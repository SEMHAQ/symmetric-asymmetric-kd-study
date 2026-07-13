## 6. Discussion

### 6.1 The Symmetry Spectrum

Our taxonomy, while conceptually useful, represents a **spectrum** rather than a strict dichotomy. Several methods exhibit hybrid characteristics:

- **Attention Transfer** computes attention maps through spatial aggregation (capturing relational structure across spatial locations—an asymmetric operation) but then directly aligns these maps between teacher and student (a symmetric loss). This dual nature suggests that the symmetric/asymmetric distinction operates at multiple levels: the **representation transformation** (how features are processed before comparison) and the **comparison operation** (how the processed representations are aligned).

- **FitNet** uses a learned regression function $r(\cdot)$ to map student features to the teacher's feature space. While the loss is symmetric (L2 comparison), the learned mapping introduces an asymmetric component—the student is not required to produce features that are directly in the same space as the teacher; an intermediate transformation is learned.

- **KD with temperature scaling** softens the probability distribution before comparison. While the KL divergence is symmetric in distribution space, the temperature scaling transforms the distributions in a way that emphasizes relative relationships between classes, introducing an asymmetric element.

In practice, most KD methods lie somewhere on the symmetric-asymmetric continuum. We recommend that researchers clearly specify **which level** (representation transformation vs. comparison operation) they are referring to when classifying a method as symmetric or asymmetric.

### 6.2 Why Symmetric Methods Converge Faster

The faster convergence of symmetric KD methods in early training can be understood through the lens of optimization geometry. Symmetric methods provide **per-sample gradient signals** that directly push each student feature toward its teacher counterpart. This creates a dense, informative gradient field. In contrast, asymmetric methods provide gradients through relational structures that depend on pairwise interactions, resulting in sparser gradient signals. Symmetric methods provide **stronger per-sample supervision** in the early stages when feature representations are still forming.

From an invariance perspective, symmetric KD explicitly enforces that the student's feature map is **equivariant** with respect to the teacher's feature map under the same input transformation. Asymmetric KD, by preserving relational structures, enforces **invariance** properties. Equivariance constraints are stronger and provide more per-sample gradient information, explaining faster convergence.

### 6.3 Why Asymmetric Methods Can Achieve Higher Peak Performance

The superior peak performance of SP (77.85%) can be attributed to the **flexibility** afforded by relational knowledge transfer. By preserving similarity structures rather than forcing direct feature alignment, SP allows the student to develop representations naturally suited to its capacity while maintaining the teacher's discriminative structure. This is particularly important when the student has fewer parameters (11.2M vs. 21.3M), as direct feature alignment may force the student to represent information in ways that exceed its representational capacity.

However, we caution against over-interpreting small accuracy differences. The gap between KD (77.71%) and SP (77.85%) is only 0.14 percentage points. Without multiple runs, it is unclear whether this difference is statistically significant. The more robust finding is that **both** strong symmetric and asymmetric methods (KD, AT, SP) reliably outperform the baseline and the teacher.

### 6.4 Guidelines for Method Selection

Based on our findings, we offer preliminary observations:

| Scenario | Suggested Approach | Rationale |
|:---|---:|:---|
| Limited training budget | Symmetric (KD or AT) | Faster convergence observed |
| Hyperparameter sensitivity concern | Symmetric (KD) | Fewer hyperparameters, more stable |
| High peak accuracy target | Asymmetric (SP) | Highest overall accuracy in our study |

These guidelines are tentative and based on a single experimental setup. Validation across more diverse settings is needed.

### 6.5 Limitations

This study has several limitations that should be acknowledged:

1. **Single dataset and architecture.** All experiments are on CIFAR-100 with ResNet variants. Generalizability to ImageNet-scale datasets and other architectures (e.g., Transformers) is not established.

2. **Single-run experiments.** Results are from single runs per method. Multi-run experiments with statistical significance testing would strengthen the conclusions.

3. **Hyperparameter borrowing.** Hyperparameters were adopted from prior literature without per-method tuning, which may disadvantage some methods (particularly FitNet and CC, which underperformed the baseline).

4. **Limited method coverage.** While we include six methods, several important approaches (e.g., Contrastive Representation Distillation [18], Knowledge Review) were not included due to computational constraints.

5. **Taxonomy ambiguity.** As discussed in Section 6.1, the boundary between symmetric and asymmetric methods is not always sharp. The taxonomy should be understood as a heuristic lens rather than a formal classification.

## 7. Conclusion

In this paper, we introduced a perspective for understanding knowledge distillation methods through the lens of symmetry. By categorizing methods as symmetric (direct representation alignment) or asymmetric (relational structure preservation), we provide a complementary framework to existing taxonomies. Our experimental comparison on CIFAR-100 with six KD methods reveals that symmetric methods converge faster while asymmetric methods can achieve competitive or higher peak performance. The best-performing method, Similarity Preserving (SP), achieves 77.85% accuracy, surpassing both the baseline and the teacher network. We discuss boundary cases, acknowledge the spectrum-like nature of the taxonomy, and identify important limitations. This work aims to provide practitioners with a useful conceptual tool for understanding and selecting knowledge distillation strategies.
