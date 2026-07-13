# Domain Review Report

**Reviewer:** Peer Reviewer 2 (Knowledge Distillation Specialist)
**Paper:** Symmetric versus Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Domain Contribution: 6/10

### Strengths
- The symmetric/asymmetric framing offers a fresh organizational lens for understanding KD methods
- Controlled comparison with six methods under unified conditions is well-executed
- The "symmetry spectrum" discussion (Section 6.1) thoughtfully addresses boundary cases

### Weaknesses

**W1. Novelty relative to existing taxonomies (MAJOR).** The field already has well-established distinctions: output-level vs. feature-level [Romero+ 2015], logit-based vs. structure-based [Gou et al. 2021], instance-level vs. relational-level [Park+ 2019]. The symmetric/asymmetric framing maps closely onto "instance-level vs. relational-level." The paper should articulate what new insight this relabeling provides. A taxonomy is valuable if it enables new predictions or method designs—the paper does not demonstrate this.

**W2. Missing key methods (MAJOR).** Several influential methods are absent:
- Contrastive Representation Distillation (CRD) [Tian+ 2020]—widely cited, structurally asymmetric
- Knowledge Review [Chen+ 2021]—multi-level feature alignment
- Self-distillation methods (e.g., Born-Again Networks [Furlanello+ 2018])
Including at least CRD would strengthen the comparison.

**W3. Limited theoretical analysis (MINOR).** The paper makes intuitive arguments about why symmetric/asymmetric methods behave differently but provides no formal analysis. For example: can the convergence rate difference be characterized in terms of the optimization landscape? Is there an information-theoretic interpretation?

### Missing References
- Gou, J.; Yu, B.; Maybank, S.J.; Tao, D. Knowledge distillation: A survey. *IJCV* 2021.
- Furlanello, T.; Lipton, Z.C.; Tschannen, M.; Itti, L.; Anandkumar, A. Born-again neural networks. *ICML* 2018.

## Recommendation: Major Revision

Address novelty differentiation from existing taxonomies and expand method coverage. The paper would be stronger as "A Symmetry-Based Perspective on Knowledge Distillation" with a clearer articulation of what this view enables that prior taxonomies do not.
