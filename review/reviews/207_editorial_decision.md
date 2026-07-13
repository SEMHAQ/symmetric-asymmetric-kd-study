# Editorial Decision Letter

**Journal:** Symmetry (MDPI)
**Manuscript ID:** SYMMETRY-2026-XXXXX
**Title:** Symmetric versus Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Decision: Major Revision

---

## Editorial Summary

Dear Authors,

Thank you for submitting your manuscript to Symmetry. The paper has been reviewed by the Editor-in-Chief and three peer reviewers, along with a Devil's Advocate assessment. All reviewers acknowledge the potential utility of a symmetry-based perspective on knowledge distillation methods. However, several substantive issues must be addressed.

---

## Consensus Points (All Reviewers)

| Severity | Issue | Source |
|:--------:|:------|:-------|
| 🔴 CRITICAL | Single-run experiments; no statistical significance | Methodology, Devil's Advocate |
| 🔴 CRITICAL | Superficial symmetry engagement—must deepen for Symmetry journal | EIC, Perspective |
| 🟡 MAJOR | Novelty vs. existing taxonomies not clearly differentiated | Domain, Devil's Advocate |
| 🟡 MAJOR | Missing methods (CRD, Knowledge Review) weaken comparison | Domain |
| 🟡 MAJOR | No testable predictions derived from the taxonomy | Devil's Advocate |
| 🟢 MINOR | Practical guidelines are qualitative, not quantitative | Perspective |
| 🟢 MINOR | Only top-1 accuracy reported | Methodology |

---

## Required Revisions (for Resubmission)

### Must Address
1. **Statistical validation.** Add multi-run experiments (3 seeds) for top-3 methods (KD, AT, SP), or clearly frame as preliminary study with appropriate caveats.
2. **Deepen symmetry analysis.** Add a formal characterization: symmetric KD = enforcing equivariance; asymmetric KD = preserving invariance. Connect to mathematical symmetry concepts.
3. **Differentiate from existing taxonomies.** Explicitly state what the symmetric/asymmetric lens offers that "instance-level vs. relational-level" does not.
4. **Acknowledge taxonomic limitations.** The framework is descriptive, not predictive. Add a subsection on "What the taxonomy does and does not tell us."

### Strongly Recommended
5. Add at least one additional method (CRD) to expand coverage.
6. Include computational cost comparison (training time per method).
7. Add ablation on batch size for SP to test the Devil's Advocate's alternative explanation.

---

## Reviewer Scores Summary

| Reviewer | Score | Recommendation |
|:---------|:----:|:--------------|
| Editor-in-Chief | 6/10 | Major Revision |
| Methodology Reviewer | 5/10 | Major Revision |
| Domain Reviewer | 6/10 | Major Revision |
| Perspective Reviewer | 5/10 | Major Revision |
| Devil's Advocate | — | Major Revision |

---

## Next Steps

Please submit a revised manuscript addressing all CRITICAL and MAJOR issues, accompanied by a point-by-point response to each reviewer comment. The revised manuscript will be subject to re-review.

*Editor-in-Chief, Symmetry (MDPI)*
