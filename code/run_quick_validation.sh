#!/bin/bash
# Quick validation pipeline: teacher → baseline → KD methods
# Run from the code/ directory

set -e

echo "============================================"
echo "  Symmetry Paper — Quick Validation Pipeline"
echo "  CIFAR-100 | ResNet-34 → ResNet-18"
echo "  60 epochs each"
echo "============================================"

export PYTHONUNBUFFERED=1

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

STEP=0

# Step 1: Train teacher
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: Teacher ResNet-34 ---"
python3 train_teacher.py --epochs 60

# Step 2: Student baseline
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: Student Baseline (no KD) ---"
python3 train_student.py --method baseline --epochs 60

# Step 3: KD (Hinton) — Symmetric
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: KD (Hinton) — Symmetric ---"
python3 train_student.py --method kd --epochs 60 --alpha 0.5 --temperature 4.0

# Step 4: FitNet — Symmetric
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: FitNet — Symmetric ---"
python3 train_student.py --method fitnet --epochs 60 --beta 1000

# Step 5: AT — Symmetric
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: AT (Attention Transfer) — Symmetric ---"
python3 train_student.py --method at --epochs 60 --beta 1000

# Step 6: SP — Asymmetric
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: SP (Similarity Preserving) — Asymmetric ---"
python3 train_student.py --method sp --epochs 60 --beta 3000

# Step 7: CC — Asymmetric
STEP=$((STEP+1))
echo ""
echo "--- Step $STEP/7: CC (Correlation Congruence) — Asymmetric ---"
python3 train_student.py --method cc --epochs 60 --beta 0.02 --gamma 0.4

echo ""
echo "============================================"
echo "  Quick validation complete!"
echo "  Results in: ../experiments/results/"
echo "============================================"
