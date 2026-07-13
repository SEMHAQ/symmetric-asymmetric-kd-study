## 1. Introduction

Deep convolutional neural networks (CNNs) have achieved remarkable success in image classification tasks, with architectures such as ResNet [1], DenseNet [2], and EfficientNet [3] pushing the boundaries of accuracy. However, the superior performance of these models comes at the cost of substantial computational and memory requirements, making their deployment on resource-constrained devices challenging [4]. This trade-off between performance and efficiency has motivated extensive research on model compression techniques, including pruning [5], quantization [6], and knowledge distillation [7].

Knowledge distillation (KD), introduced by Hinton et al. [7], addresses this challenge by transferring knowledge from a large, pre-trained teacher network to a smaller student network. The student is trained to mimic the teacher's output distribution, typically through minimizing the Kullback-Leibler (KL) divergence between their softened probability distributions. Since this seminal work, numerous extensions have been proposed, which can be broadly categorized along multiple dimensions: output-level vs. feature-level [8,9], logit-based vs. structure-based [10,11], and instance-level vs. relational-level [12].

In this paper, we propose a complementary perspective that cuts across these existing categorizations: **the symmetry of the knowledge transfer mechanism**. Specifically, some KD methods perform a direct, one-to-one alignment between teacher and student representations. We term these **symmetric** methods, as they assume a symmetric correspondence between the representation spaces of the two networks—the student should learn to produce features that directly match the teacher's features on a per-sample, per-feature basis. Examples include the original KD (output distribution matching), FitNet [8] (intermediate feature regression), and Attention Transfer (AT) [9] (attention map alignment).

In contrast, other methods preserve the relational structure among samples without requiring direct feature correspondence. We term these **asymmetric** methods, as the knowledge is encoded in the relationships between entities rather than in direct mappings between teacher and student representation spaces. Examples include Similarity Preserving (SP) [10], Correlation Congruence (CC) [11], and Relational Knowledge Distillation (RKD) [12].

This distinction has practical implications. When teacher and student networks share similar architectures, their intermediate representations are likely to exhibit similar structures, making symmetric alignment natural and effective. However, when the architecture gap is large (e.g., a deep teacher vs. a shallow student), direct feature alignment may force the student to learn representations that are not naturally suited to its capacity. Asymmetric methods, by focusing on relational structures, may be more robust to such architectural disparities. We note, however, that this taxonomy represents a spectrum rather than a strict dichotomy—some methods incorporate both symmetric and asymmetric components, as we discuss in Section 6.

In this paper, we present a systematic comparative study of symmetric and asymmetric KD methods under unified experimental conditions. Our main contributions are:

1. **A new perspective** for understanding knowledge distillation methods based on the symmetry of their knowledge transfer mechanisms, providing a complementary lens to existing taxonomies.

2. **A comprehensive empirical comparison** of six representative KD methods (KD, FitNet, AT, SP, CC, RKD) on the CIFAR-100 dataset, using a standardized ResNet-34 → ResNet-18 teacher-student setup with identical training protocols.

3. **Practical observations** on when symmetric vs. asymmetric distillation may be preferred, along with discussion of limitations and boundary cases.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 formally defines the taxonomy. Section 4 describes the experimental setup. Section 5 presents results. Section 6 discusses implications and limitations. Section 7 concludes the paper.
