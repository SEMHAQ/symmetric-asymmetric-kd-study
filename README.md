# Symmetric versus Asymmetric Knowledge Distillation

> **All experiment code is open source and fully reproducible.** 🚀

A comparative study of symmetric and asymmetric knowledge distillation methods for image classification, published in **Symmetry** (MDPI).

## 🔑 Highlights

- **Open-source implementation** of 6 KD methods (KD, FitNet, AT, SP, CC, RKD) under a unified training framework
- **Single-command reproduction** — train teacher, run any KD method, generate all figures
- **Real experimental data** — all results from actual model training on CIFAR-100 with ResNet-34 → ResNet-18
- **Per-class analysis** — confusion-matrix-based evaluation across 100 classes

## 📊 Key Results

| Method | Category | Accuracy | Δ vs Baseline |
|--------|----------|:--------:|:-------------:|
| **SP** | Asymmetric | **77.85%** | **+1.37%** |
| KD | Symmetric | 77.71% | +1.23% |
| AT | Symmetric | 77.34% | +0.86% |
| Baseline | — | 76.48% | — |

**Setup:** CIFAR-100, ResNet-34 teacher (76.75%) → ResNet-18 student, 60 epochs, cosine LR.

## 🚀 Quick Start

### Install

```bash
pip install torch torchvision matplotlib seaborn numpy tqdm
```

### Train Teacher

```bash
cd code
python train_teacher.py --epochs 60
```

### Train Student with Any KD Method

```bash
# Symmetric methods
python train_student.py --method kd     --epochs 60 --alpha 0.5 --temperature 4.0
python train_student.py --method fitnet  --epochs 60 --beta 1000
python train_student.py --method at      --epochs 60 --beta 500

# Asymmetric methods
python train_student.py --method sp      --epochs 60 --beta 3000
python train_student.py --method cc      --epochs 60 --beta 0.02 --gamma 0.4
python train_student.py --method rkdi    --epochs 60 --beta 1.0
```

### Generate Figures

```bash
python analyze_results.py
```

## 📁 Project Structure

```
├── code/                        # All experiment code
│   ├── train_teacher.py         # Train ResNet-34 teacher
│   ├── train_student.py         # Train ResNet-18 with any KD method
│   ├── analyze_results.py       # Generate comparison figures
│   ├── kd_methods/losses.py     # KD loss implementations
│   └── models/resnet_cifar.py   # CIFAR-adapted ResNet
├── experiments/results/         # Training logs (JSON)
└── README.md
```

## 📝 Citation

If you use this code, please cite our paper:

```bibtex
@article{peng2026symmetry,
  title={A Symmetry-Based Perspective on Knowledge Distillation: Symmetric versus Asymmetric Approaches for Image Classification},
  author={Peng, Donghai and Yu, Huanjie},
  journal={Symmetry},
  year={2026},
  publisher={MDPI}
}
```

## 📄 License

MIT
