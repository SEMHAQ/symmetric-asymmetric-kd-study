"""Train student with various KD methods.

Usage:
  # Baseline (no KD)
  python train_student.py --method baseline

  # Symmetric KD methods
  python train_student.py --method kd      --alpha 0.5 --temperature 4.0
  python train_student.py --method fitnet  --beta 1000
  python train_student.py --method at      --beta 1000

  # Asymmetric KD methods
  python train_student.py --method sp      --beta 3000
  python train_student.py --method cc      --beta 0.02 --gamma 0.4
  python train_student.py --method rkdi    --beta 1.0

Result: saves checkpoint + metrics JSON to experiments/results/KD_{method}/
"""
import os
import sys
import json
import time
import math
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET, NUM_CLASSES, DATA_ROOT, DEVICE, NUM_WORKERS, PIN_MEMORY,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
    TEACHER_ARCH, TEACHER_CKPT, STUDENT_ARCH,
    KD_TEMPERATURE, KD_ALPHA, FITNET_BETA, AT_BETA, SP_BETA, CC_BETA, CC_GAMMA,
    SACD_LAMBDA,
)
from models.resnet_cifar import resnet18, resnet34
from kd_methods.losses import get_kd_loss, KD_METHODS


def get_dataloaders(batch_size=128):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR100(root=DATA_ROOT, train=True,
                                  download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root=DATA_ROOT, train=False,
                                  download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, test_loader


def get_feature_maps(model, x, target_layers=None):
    """Forward pass that returns intermediate features.

    For ResNet-CIFAR, the relevant layers are layer2, layer3, layer4.
    We hook into the forward pass to get feature maps.
    """
    # Manual feature extraction for our ResNet
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out2 = model.layer2(out)
    out3 = model.layer3(out2)
    out4 = model.layer4(out3)
    pooled = model.avgpool(out4)
    pooled_flat = pooled.view(pooled.size(0), -1)
    logits = model.linear(pooled_flat)
    return logits, {
        'layer2': out2,
        'layer3': out3,
        'layer4': out4,
        'avgpool': pooled_flat,
    }


def get_hint_maps(teacher, student, x):
    """Get aligned feature maps for FitNet."""
    with torch.no_grad():
        # Teacher features at layer2 and layer3
        _, t_feats = get_feature_maps(teacher, x)
        t_hint = t_feats['layer3']  # (N, 256, 8, 8) for ResNet-34

    _, s_feats = get_feature_maps(student, x)
    s_guided = s_feats['layer2']  # (N, 128, 8, 8) for ResNet-18

    return s_guided, t_hint


def get_attention_maps(teacher, student, x):
    """Get attention maps for AT loss."""
    with torch.no_grad():
        _, t_feats = get_feature_maps(teacher, x)
        t_attn = t_feats['layer4']

    _, s_feats = get_feature_maps(student, x)
    s_attn = s_feats['layer4']
    return s_attn, t_attn


def get_features_for_kd(teacher, student, x):
    """Get pooled features for relational KD methods (SP, CC, RKD)."""
    with torch.no_grad():
        _, t_feats = get_feature_maps(teacher, x)
        t_feat = t_feats['avgpool']

    _, s_feats = get_feature_maps(student, x)
    s_feat = s_feats['avgpool']
    return s_feat, t_feat


def train_epoch_kd(model, teacher, loader, optimizer, args, epoch=0, total_epochs=60):
    """Training epoch with KD loss."""
    model.train()
    if teacher is not None:
        teacher.eval()

    running_loss = 0.0
    running_kd_loss = 0.0
    running_ce_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        if args.method == 'baseline':
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            kd_loss_val = 0.0
            ce_loss_val = loss.item()

        elif args.method == 'kd':
            outputs = model(inputs)
            with torch.no_grad():
                t_outputs = teacher(inputs)
            ce_loss = F.cross_entropy(outputs, targets)
            kd_loss = F.kl_div(
                F.log_softmax(outputs / args.temperature, dim=1),
                F.softmax(t_outputs / args.temperature, dim=1),
                reduction='batchmean'
            ) * (args.temperature ** 2)
            loss = args.alpha * kd_loss + (1 - args.alpha) * ce_loss
            kd_loss_val = kd_loss.item()
            ce_loss_val = ce_loss.item()

        elif args.method in ('fitnet',):
            outputs, s_feats = get_feature_maps(model, inputs)
            with torch.no_grad():
                _, t_feats = get_feature_maps(teacher, inputs)

            s_feat = s_feats['avgpool']  # (N, 512)
            t_feat = t_feats['avgpool']  # (N, 512)
            ce_loss = F.cross_entropy(outputs, targets)
            fitnet_loss = F.mse_loss(s_feat, t_feat) * args.beta
            loss = ce_loss + fitnet_loss
            kd_loss_val = fitnet_loss.item()
            ce_loss_val = ce_loss.item()

        elif args.method == 'at':
            outputs = model(inputs)
            s_feat, t_feat = get_features_for_kd(teacher, model, inputs)
            ce_loss = F.cross_entropy(outputs, targets)

            # AT loss on pooled features: normalized activation magnitude
            def _attn_map(x):
                norm = x.norm(p=2, dim=1, keepdim=True) + 1e-8
                return x / norm

            at_loss = F.mse_loss(_attn_map(s_feat), _attn_map(t_feat)) * args.beta
            loss = ce_loss + at_loss
            kd_loss_val = at_loss.item()
            ce_loss_val = ce_loss.item()

        elif args.method in ('sp', 'cc', 'rkdi'):
            outputs = model(inputs)
            s_feat, t_feat = get_features_for_kd(teacher, model, inputs)
            ce_loss = F.cross_entropy(outputs, targets)

            if args.method == 'sp':
                # Similarity Preserving
                def _similarity(f):
                    f = f.view(f.size(0), -1)
                    f_norm = F.normalize(f, p=2, dim=1)
                    return f_norm @ f_norm.T

                G_s = _similarity(s_feat)
                G_t = _similarity(t_feat)
                G_s = G_s / G_s.norm(p=2, dim=1, keepdim=True)
                G_t = G_t / G_t.norm(p=2, dim=1, keepdim=True)
                kd_loss = F.mse_loss(G_s, G_t) * args.beta

            elif args.method == 'cc':
                # Correlation Congruence
                def _gram(x):
                    x = x.view(x.size(0), -1)
                    x = x - x.mean(dim=0, keepdim=True)
                    gram = x.T @ x / (x.size(0) - 1)
                    gram = gram / (gram.norm(p='fro') + 1e-8)
                    return gram

                gram_s = _gram(s_feat)
                gram_t = _gram(t_feat)
                kd_loss = F.mse_loss(gram_s, gram_t) * args.beta

            elif args.method == 'rkdi':
                # Relational KD - distance-wise
                def _pair_dist(x):
                    x = x.view(x.size(0), -1)
                    x_norm = x.pow(2).sum(dim=1, keepdim=True)
                    dist = x_norm + x_norm.T - 2.0 * (x @ x.T)
                    dist = F.relu(dist).sqrt()
                    dist = dist / (dist.mean(dim=1, keepdim=True) + 1e-8)
                    return dist

                d_s = _pair_dist(s_feat)
                d_t = _pair_dist(t_feat)
                kd_loss = F.mse_loss(d_s, d_t) * args.beta

            loss = ce_loss + kd_loss
            kd_loss_val = kd_loss.item()
            ce_loss_val = ce_loss.item()

        elif args.method == 'sacd':
            outputs, s_feats = get_feature_maps(model, inputs)
            with torch.no_grad():
                _, t_feats = get_feature_maps(teacher, inputs)
            ce_loss = F.cross_entropy(outputs, targets)

            # SACD: per-layer adaptive symmetric/asymmetric distillation
            layers = ['layer2', 'layer3', 'layer4']
            total_kd = 0.0
            n_layers = 0

            for layer_name in layers:
                s_f = s_feats[layer_name]  # (N, C, H, W)
                t_f = t_feats[layer_name]

                # Adaptive pool to same spatial size
                if s_f.shape[2:] != t_f.shape[2:]:
                    min_h = min(s_f.shape[2], t_f.shape[2])
                    min_w = min(s_f.shape[3], t_f.shape[3])
                    s_f = F.adaptive_avg_pool2d(s_f, (min_h, min_w))
                    t_f = F.adaptive_avg_pool2d(t_f, (min_h, min_w))

                # Flatten to (N, C*H*W)
                s_flat = s_f.view(s_f.size(0), -1)
                t_flat = t_f.view(t_f.size(0), -1)

                # Adaptive weight β_l based on feature similarity
                s_norm = F.normalize(s_flat, p=2, dim=1)
                t_norm = F.normalize(t_flat, p=2, dim=1)
                cos_sim = (s_norm * t_norm).sum(dim=1).mean()
                # β_l ∈ [0.3, 0.8]: high cos_sim → more symmetric (direct alignment)
                beta_l = 0.3 + 0.5 * torch.sigmoid(cos_sim)

                # Symmetric loss: MSE on raw (unnormalized) features
                # Scale-invariant by dividing by per-channel std
                s_std = s_flat.std(dim=0, keepdim=True) + 1e-8
                t_std = t_flat.std(dim=0, keepdim=True) + 1e-8
                l_sym = F.mse_loss(s_flat / s_std, t_flat / t_std)

                # Asymmetric loss: SP (similarity preserving)
                G_s = s_norm @ s_norm.T
                G_t = t_norm @ t_norm.T
                G_s = G_s / (G_s.norm(p=2, dim=1, keepdim=True) + 1e-8)
                G_t = G_t / (G_t.norm(p=2, dim=1, keepdim=True) + 1e-8)
                l_asym = F.mse_loss(G_s, G_t)

                # Weighted combination
                total_kd = total_kd + beta_l * l_sym + (1 - beta_l) * l_asym
                n_layers += 1

            kd_loss = total_kd / n_layers
            # Cosine decay: λ starts at args.lmbda, decays to 0.05*args.lmbda
            progress = epoch / total_epochs
            lmbda_t = args.lmbda * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
            loss = ce_loss + lmbda_t * kd_loss
            kd_loss_val = kd_loss.item()
            ce_loss_val = ce_loss.item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_kd_loss += kd_loss_val * inputs.size(0)
        running_ce_loss += ce_loss_val * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(loader)} | '
                  f'Loss: {loss.item():.4f} | CE: {ce_loss_val:.4f} | '
                  f'KD: {kd_loss_val:.4f}')

    n = total
    return (running_loss / n, running_ce_loss / n, running_kd_loss / n,
            100.0 * correct / total)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        losses.append(loss.item() * inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = sum(losses) / total
    return avg_loss, acc


def main(args):
    print(f'\n{"="*60}')
    print(f'[Student] Training with method: {args.method.upper()}')
    print(f'{"="*60}')

    # Teacher
    if args.method != 'baseline':
        print(f'\nLoading teacher ({TEACHER_ARCH}) from {TEACHER_CKPT}')
        teacher = resnet34(num_classes=NUM_CLASSES, return_features=False)
        checkpoint = torch.load(TEACHER_CKPT, map_location=DEVICE)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher = teacher.to(DEVICE)
        teacher.eval()
        teacher_acc = checkpoint['acc']
        print(f'  Teacher accuracy: {teacher_acc:.2f}%')
    else:
        teacher = None
        teacher_acc = None

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Student model
    model = resnet18(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(f'  Student params: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_ce': [], 'train_kd': [], 'train_acc': [],
        'test_acc': [],
        'method': args.method,
        'params': vars(args),
        'teacher_acc': teacher_acc,
    }
    output_dir = f'../experiments/results/KD_{args.method.upper()}'
    ckpt_path = f'{output_dir}/student_{args.method}_best.pth'
    os.makedirs(output_dir, exist_ok=True)

    print(f'\nTraining {args.epochs} epochs...\n')
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_ce, train_kd, train_acc = train_epoch_kd(
            model, teacher, train_loader, optimizer, args,
            epoch=epoch, total_epochs=args.epochs
        )
        test_loss, test_acc = evaluate(model, test_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_ce'].append(train_ce)
        history['train_kd'].append(train_kd)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        elapsed = time.time() - start
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}% | '
              f'{elapsed:.0f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'method': args.method,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': test_acc,
            }, ckpt_path)
            print(f'  ✓ New best: {test_acc:.2f}% → saved')

    print(f'\n🏆 [{args.method.upper()}] Best test accuracy: {best_acc:.2f}%')

    # Save history
    history['best_acc'] = best_acc
    with open(f'{output_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Results saved to {output_dir}/')

    return best_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['baseline', 'kd', 'fitnet', 'at',
                                 'sp', 'cc', 'rkdi', 'sacd'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)

    # KD-specific hyperparameters
    parser.add_argument('--temperature', type=float, default=KD_TEMPERATURE)
    parser.add_argument('--alpha', type=float, default=KD_ALPHA)
    parser.add_argument('--beta', type=float, default=None)  # per-method default
    parser.add_argument('--gamma', type=float, default=CC_GAMMA)
    parser.add_argument('--lmbda', type=float, default=None)  # SACD overall weight
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set per-method beta defaults
    beta_defaults = {
        'fitnet': FITNET_BETA,
        'at': AT_BETA,
        'sp': SP_BETA,
        'cc': CC_BETA,
        'rkdi': 1.0,
    }
    if args.beta is None:
        args.beta = beta_defaults.get(args.method, 0)
    if args.lmbda is None:
        args.lmbda = SACD_LAMBDA

    main(args)
