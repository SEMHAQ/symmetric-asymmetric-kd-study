"""Train teacher model (ResNet-34 on CIFAR-100)."""
import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET, NUM_CLASSES, DATA_ROOT, DEVICE, NUM_WORKERS, PIN_MEMORY,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
    TEACHER_ARCH, TEACHER_CKPT
)
from models.resnet_cifar import resnet34


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


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}')

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


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
    print(f'[Teacher] Training {TEACHER_ARCH} on {DATASET}')
    print(f'  Device: {DEVICE}')
    print(f'  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}')

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Model
    model = resnet34(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, epoch)
        test_loss, test_acc = evaluate(model, test_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        elapsed = time.time() - start
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}% | '
              f'LR: {scheduler.get_last_lr()[0]:.5f} | '
              f'{elapsed:.0f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(TEACHER_CKPT), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': test_acc,
            }, TEACHER_CKPT)
            print(f'  ✓ New best: {test_acc:.2f}% → saved')

    print(f'\n🏆 Teacher best test accuracy: {best_acc:.2f}%')

    # Save history
    results_dir = os.path.dirname(TEACHER_CKPT)
    with open(os.path.join(results_dir, 'teacher_history.json'), 'w') as f:
        json.dump(history, f)
    print(f'History saved to {results_dir}/teacher_history.json')

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args)
