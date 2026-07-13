## 2. Related Work

### 2.1 Knowledge Distillation

Knowledge distillation, originally formulated by Hinton et al. [7], trains a student network to match the softened output distribution of a teacher network. The student loss combines the standard cross-entropy with the ground-truth labels and the KL divergence with respect to the teacher's softened predictions:

$$L_{KD} = \alpha \cdot L_{CE}(y, \sigma(z_s)) + (1-\alpha) \cdot \tau^2 \cdot KL(\sigma(z_t/\tau), \sigma(z_s/\tau))$$

where $z_s$ and $z_t$ are the student and teacher logits, $\tau$ is the temperature parameter controlling the softness of the probability distribution, $\sigma$ denotes the softmax function, and $\alpha$ balances the two loss terms.

Since Hinton's formulation, numerous extensions have been proposed. Romero et al. [8] introduced FitNet, which adds an L2 loss between intermediate feature representations of the teacher and student, enabling the student to learn not only the final predictions but also the intermediate representations. Zagoruyko and Komodakis [9] proposed Attention Transfer (AT), which aligns the spatial attention maps derived from teacher and student feature maps. These methods share a common characteristic: they directly match specific representations between teacher and student.

A different line of research focuses on preserving relational structures. Park et al. [12] introduced Relational Knowledge Distillation (RKD), which preserves pairwise and higher-order relationships among samples in the teacher's embedding space. Tung and Mori [10] proposed Similarity Preserving (SP) distillation, which preserves the pairwise similarity matrix of samples within a mini-batch. Peng et al. [11] introduced Correlation Congruence (CC), which aligns the second-order feature correlations between teacher and student. These methods do not require direct one-to-one correspondence between representations; instead, they preserve the structural relationships among samples.

### 2.2 Symmetry in Deep Learning

The concept of symmetry has deep roots in deep learning, appearing in various forms. Convolutional neural networks inherently exploit translational symmetry through weight sharing [13]. Group-equivariant CNNs [14] extend this to broader symmetry groups, including rotations and reflections. Self-attention mechanisms in Transformers [15] exhibit permutation symmetry. In the context of representation learning, contrasting symmetric and asymmetric learning paradigms has proven fruitful: SimSiam [16] and BYOL [17] demonstrate that asymmetric architectures can prevent representational collapse in self-supervised learning.

Our work draws a parallel distinction for knowledge distillation, categorizing methods based on whether they employ symmetric (direct alignment) or asymmetric (relational preservation) knowledge transfer mechanisms.

### 2.3 Comparative Studies of KD Methods

Several prior works have compared KD methods empirically. Tian et al. [18] evaluated multiple KD methods on large-scale benchmarks, revealing that simple KL-divergence-based KD remains competitive when properly tuned. However, these studies typically focus on performance rankings rather than providing a conceptual framework for understanding when and why different methods excel. Our work fills this gap by introducing the symmetry/asymmetry taxonomy and systematically evaluating its predictive power for method selection.
