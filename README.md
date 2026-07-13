# Symmetric vs. Asymmetric Knowledge Distillation

A comparative study of symmetric and asymmetric knowledge distillation methods for image classification, targeting **Symmetry** (MDPI) Special Issue on *Symmetry and Asymmetry on Artificial Neural Networks for Visual Learning*.

---

## 📄 Paper

| File | Description |
|------|-------------|
| [`manuscript/manuscript.tex`](manuscript/manuscript.tex) | LaTeX source (MDPI template) |
| [`manuscript/manuscript.pdf`](manuscript/manuscript.pdf) | Compiled PDF (8 pages) |
| [`manuscript/figures/`](manuscript/figures/) | Vector figures (PDF) |

## 🔬 Methods Compared

| Method | Category | Accuracy | Δ vs Baseline |
|--------|----------|:--------:|:-------------:|
| **SP (Tung & Mori, 2019)** | Asymmetric | **77.85%** | **+1.37%** |
| KD (Hinton et al., 2015) | Symmetric | 77.71% | +1.23% |
| AT (Zagoruyko & Komodakis, 2017) | Symmetric | 77.34% | +0.86% |
| RKD (Park et al., 2019) | Asymmetric | 76.43% | -0.05% |
| Baseline (No KD) | — | 76.48% | — |
| FitNet (Romero et al., 2015) | Symmetric | 76.06% | -0.42% |
| CC (Peng et al., 2020) | Asymmetric | 75.81% | -0.67% |

**Setup:** CIFAR-100, Teacher: ResNet-34 (76.75%), Student: ResNet-18, 60 epochs.

## 🚀 Reproduce Experiments

### Prerequisites

- Python 3.10+
- PyTorch 2.1+ with CUDA
- torchvision, numpy, matplotlib, seaborn, tqdm

### Train Teacher

```bash
cd code
python3 train_teacher.py --epochs 60
```

### Train Student with KD

```bash
# Baseline
python3 train_student.py --method baseline --epochs 60

# Symmetric methods
python3 train_student.py --method kd --epochs 60 --alpha 0.5 --temperature 4.0
python3 train_student.py --method fitnet --epochs 60 --beta 1000
python3 train_student.py --method at --epochs 60 --beta 500

# Asymmetric methods
python3 train_student.py --method sp --epochs 60 --beta 3000
python3 train_student.py --method cc --epochs 60 --beta 0.02 --gamma 0.4
python3 train_student.py --method rkdi --epochs 60 --beta 1.0
```

### Analyze Results

```bash
python3 analyze_results.py
```

Output: Comparison tables + 3 figures (`experiments/figures/`).

## 📊 Results

All experiment logs and metrics are stored as JSON in [`experiments/results/`](experiments/results/).

## 📁 Structure

```
├── manuscript/              # Paper (LaTeX + PDF + figures)
├── code/                    # Experiment code
│   ├── train_teacher.py     # Teacher training
│   ├── train_student.py     # KD training (all methods)
│   ├── analyze_results.py   # Results analysis & figures
│   ├── kd_methods/          # KD loss implementations
│   └── models/              # Network architectures
├── experiments/results/     # Numerical results
├── review/                  # Simulated peer reviews
└── README.md
```

## 📝 License

MIT
