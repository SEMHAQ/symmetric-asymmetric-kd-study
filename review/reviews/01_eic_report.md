# Editor-in-Chief Review Report

**Reviewer:** Editor-in-Chief, Symmetry (MDPI)
**Field:** Computer Vision, Knowledge Distillation
**Paper:** Symmetric vs. Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Overall Assessment

This paper presents a taxonomy of knowledge distillation methods categorized as symmetric vs. asymmetric based on their knowledge transfer mechanism. The topic is relevant to Symmetry's scope, and the comparative study is executed with proper experimental methodology.

---

## Journal Fit

**Score: 7/10**

The paper's central theme—symmetry vs. asymmetry in knowledge transfer—aligns well with Symmetry's focus. However, the connection to "symmetry" as a mathematical/physical concept is somewhat superficial. The paper uses "symmetric" and "asymmetric" as labels for method categories rather than deeply engaging with the concept of symmetry (e.g., invariance properties, group-theoretic analysis). Strengthening this connection would significantly improve journal fit.

---

## Originality

**Score: 6/10**

- The taxonomy itself (symmetric vs. asymmetric KD) is novel and has not been explicitly proposed in prior literature
- However, the individual methods compared are all existing works; the novelty lies in the classification framework, not in any new method
- The guidelines for method selection are helpful but fairly intuitive

---

## Significance

**Score: 6/10**

- Useful for practitioners who need guidance on KD method selection
- Moderate significance—confirms what some in the field might suspect but haven't systematically demonstrated
- Lacks deeper theoretical analysis of *why* symmetric/asymmetric methods differ

---

## Major Concerns

1. **Limited generalizability**: Single dataset (CIFAR-100), single architecture family (ResNet). Claims about "symmetric vs. asymmetric" broadly would require validation across more diverse settings.

2. **Hyperparameter fairness**: The paper acknowledges that hyperparameters were adopted from prior literature without tuning. This may systematically disadvantage some methods. For example, FitNet and CC underperform the baseline—is this intrinsic to the methods or due to suboptimal hyperparameters?

3. **Taxonomy boundary cases**: Some methods may not fit cleanly into either category. For instance, Attention Transfer uses attention maps derived from features (relational at the spatial level) but aligns them directly (symmetric). The paper should discuss such boundary cases.

4. **Missing statistical significance tests**: The accuracy differences between methods are small (KD 77.71% vs. SP 77.85%, Δ=0.14%). Without statistical significance testing (e.g., multiple runs with confidence intervals), these differences may not be meaningful.

---

## Recommendation

**Major Revision**

The paper has merit and is within scope for Symmetry, but requires substantial strengthening before publication. The most critical issues are: (1) deepening the symmetry connection, (2) adding statistical rigor, and (3) discussing taxonomy boundary cases.
