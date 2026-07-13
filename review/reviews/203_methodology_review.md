# Methodology Review Report

**Reviewer:** Peer Reviewer 1 (Methodology Specialist)
**Paper:** Symmetric versus Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Methodological Rigor: 5/10

### Strengths
- Clear documentation of training protocols and hyperparameters
- All methods compared under identical conditions (same epochs, optimizer, LR schedule)
- Learning curves reported alongside best accuracy

### Critical Issues

**C1. Single-run experiments (CRITICAL).** All results are from a single training run per method. Deep learning results are stochastic; without multiple runs (3-5 with different seeds), the reported accuracy differences (e.g., SP 77.85% vs. KD 77.71%) cannot be assessed for statistical significance. This is the paper's most significant methodological weakness.

**C2. No statistical testing (CRITICAL).** Claims like "symmetric methods converge faster" and "asymmetric methods achieve higher peak performance" are unsupported without confidence intervals or significance tests (e.g., paired t-test across epochs or runs).

**C3. Hyperparameter fairness (MAJOR).** Hyperparameters adopted from prior literature without per-method tuning may systematically disadvantage certain methods. FitNet (β=1000) and CC (β=0.02) underperform the baseline—is this intrinsic to these methods or an artifact of suboptimal hyperparameters?

### Minor Issues
- Only top-1 accuracy reported; top-5 or per-class metrics would enrich the comparison
- No error bars on learning curves
- Ablation on temperature τ for KD would strengthen claims about symmetric method behavior

## Reproducibility: 7/10

Experimental details are sufficiently documented. Code availability is stated. Adding exact random seeds would improve reproducibility.

## Recommendation: Major Revision

The single-run methodology must be addressed. At minimum: (1) add uncertainty quantification via bootstrapping over test set, (2) run 3 seeds for the top-3 methods to establish variance bounds, or (3) clearly frame the paper as a preliminary study and tone down comparative claims.
