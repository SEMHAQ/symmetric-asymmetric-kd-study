# Editorial Decision Letter

**Journal:** Symmetry (MDPI)
**Manuscript ID:** SYMMETRY-DRAFT-001
**Title:** Symmetric vs. Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Decision: Major Revision

---

## Editorial Summary

Dear Author,

Thank you for submitting your manuscript to Symmetry. The paper has been reviewed by the Editor-in-Chief and three peer reviewers, along with a Devil's Advocate assessment. The reviewers acknowledge the timeliness and potential utility of a systematic comparison of knowledge distillation methods. However, several substantive issues must be addressed before the manuscript can be considered for publication.

---

## Consensus Points (All Reviewers Agree)

| Issue | Severity | Description |
|:-----|:--------:|:------------|
| Single-run experiments | **CRITICAL** | All results from single runs; statistical significance cannot be assessed |
| Superficial symmetry engagement | **MAJOR** | The paper uses "symmetry" as a label without mathematical depth—critical for Symmetry journal |
| Missing methods | **MAJOR** | RKD, CRD, and Knowledge Review should be included or justified as omissions |
| Taxonomy boundary cases | **MAJOR** | Several methods blur the symmetric/asymmetric boundary; needs discussion |

---

## Editorial Requirements (Mandatory for Resubmission)

### 1. Statistical Validation (from Methodology Reviewer, Devil's Advocate)
- [ ] Run each method 3-5 times with different random seeds
- [ ] Report mean ± std for each method
- [ ] Include statistical significance tests (paired t-test or Wilcoxon) for key comparisons
- [ ] Add confidence intervals to all figures

### 2. Deepen Symmetry Analysis (from EIC, Perspective Reviewer)
- [ ] Add theoretical discussion connecting KD symmetry to invariance/equivariance
- [ ] Discuss what "symmetry" means in the context of knowledge transfer
- [ ] Consider whether the taxonomy relates to known mathematical concepts of symmetry

### 3. Strengthen Comparative Claims (from Domain Reviewer, Devil's Advocate)
- [ ] Add at least RKD to the experimental comparison
- [ ] Or rename the contribution to "instance-level vs. relational" if methods cannot be added
- [ ] Acknowledge that the taxonomy maps closely to prior dichotomies

### 4. Discuss Taxonomy Boundary Cases (from EIC, Devil's Advocate)
- [ ] Add a section discussing methods that combine symmetric and asymmetric components
- [ ] Acknowledge the spectrum nature of the taxonomy

---

## Recommended Improvements (Optional but Strongly Encouraged)

- Add ablation studies on key hyperparameters (temperature, loss weights)
- Validate on at least one additional dataset (e.g., CIFAR-10, Tiny ImageNet)
- Report additional metrics (top-5 accuracy, per-class accuracy)
- Quantify the practical guidelines in Table 2

---

## Reviewer Scores Summary

| Reviewer | Score (1-10) | Recommendation |
|:---------|:-----------:|:--------------|
| Editor-in-Chief | 6/10 | Major Revision |
| Methodology Reviewer | 6/10 | Major Revision |
| Domain Reviewer | 6/10 | Minor-Major Revision |
| Perspective Reviewer | 7/10 | Major Revision |
| Devil's Advocate | — | Major Revision |

---

## Next Steps

Please address all CRITICAL and MAJOR issues in a revised manuscript. Provide a point-by-point response to each reviewer comment. The revised manuscript will be subject to re-review.

We look forward to receiving your revised manuscript.

Sincerely,

*Editor-in-Chief*
*Symmetry (MDPI)*
