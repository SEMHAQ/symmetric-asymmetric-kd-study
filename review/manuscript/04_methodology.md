## 3. Symmetric vs. Asymmetric Knowledge Distillation

### 3.1 Formal Definition

We propose a taxonomy that categorizes knowledge distillation methods based on the symmetry of their representation alignment mechanism. Let $f_T(x) \in \mathbb{R}^{d_T}$ and $f_S(x) \in \mathbb{R}^{d_S}$ denote the feature representations of the teacher and student networks for input $x$, respectively.

**Symmetric Knowledge Distillation** minimizes a distance function $\mathcal{D}$ that directly compares teacher and student representations on a **per-sample, per-feature** basis:

$$\mathcal{L}_{sym} = \mathbb{E}_{x \sim \mathcal{X}} \left[ \mathcal{D}\big(g_T(f_T(x)), g_S(f_S(x))\big) \right]$$

where $g_T$ and $g_S$ are (optional) projection functions that map features to a common space. The key property is the **direct one-to-one correspondence** between elements of the teacher and student representations. From a symmetry perspective, symmetric KD assumes an **equivariance** property: a transformation applied to the input should produce corresponding transformations in both teacher and student features that are directly comparable.

**Asymmetric Knowledge Distillation** minimizes a distance function $\mathcal{D}$ that compares teacher and student representations **indirectly through their relational structures**:

$$\mathcal{L}_{asym} = \mathbb{E}_{(x_i, x_j) \sim \mathcal{X}^2} \left[ \mathcal{D}\big(\mathcal{R}_T(f_T(x_i), f_T(x_j)), \mathcal{R}_S(f_S(x_i), f_S(x_j))\big) \right]$$

where $\mathcal{R}$ is a relational function (e.g., pairwise similarity, distance, or correlation). The key property is the **absence of direct feature correspondence**—knowledge is transferred through the preservation of inter-sample relationships. Asymmetric KD preserves the **invariance** structure of the teacher's representation space: relationships between samples should be invariant under the mapping from teacher to student.

### 3.2 Symmetric KD Methods

**3.2.1 KD (Hinton, 2015).** The original KD method matches the softened output distributions:

$$\mathcal{L}_{KD} = KL\left(\sigma\left(\frac{z_T}{\tau}\right) \middle\| \sigma\left(\frac{z_S}{\tau}\right) \right)$$

where $z_T$ and $z_S$ are the logits. This is symmetric because it directly aligns the probability distributions for each sample.

**3.2.2 FitNet (Romero et al., 2015).** FitNet adds an L2 regression loss between intermediate features:

$$\mathcal{L}_{FitNet} = \left\| r(f_S(x)) - f_T(x) \right\|_2^2$$

where $r(\cdot)$ is a learned regression function. The direct one-to-one alignment makes this symmetric.

**3.2.3 Attention Transfer (Zagoruyko & Komodakis, 2017).** AT aligns normalized attention maps:

$$\mathcal{L}_{AT} = \left\| \frac{A(f_S(x))}{\|A(f_S(x))\|_2} - \frac{A(f_T(x))}{\|A(f_T(x))\|_2} \right\|_2^2$$

where $A(\cdot) = \sum_{c=1}^C (\cdot)^2_{c}$ computes the spatial attention map. AT is an interesting boundary case: the attention maps themselves capture relational information across spatial locations (asymmetric), but they are then directly aligned between teacher and student (symmetric). We classify AT as symmetric because the loss function directly compares teacher and student quantities, but acknowledge this hybrid nature in Section 6.

### 3.3 Asymmetric KD Methods

**3.3.1 Similarity Preserving (Tung & Mori, 2019).** SP preserves pairwise sample similarities:

$$\mathcal{L}_{SP} = \frac{1}{b^2} \left\| \frac{G_S}{\|G_S\|_F} - \frac{G_T}{\|G_T\|_F} \right\|_F^2$$

where $G_{ij} = f(x_i)^\top f(x_j)$ is the pairwise similarity matrix.

**3.3.2 Correlation Congruence (Peng et al., 2020).** CC aligns second-order feature correlations:

$$\mathcal{L}_{CC} = \left\| \frac{C_S}{\|C_S\|_F} - \frac{C_T}{\|C_T\|_F} \right\|_F^2$$

where $C = \frac{1}{b-1}(F - \bar{F})^\top(F - \bar{F})$ is the feature correlation matrix.

**3.3.3 Relational Knowledge Distillation (Park et al., 2019).** RKD preserves pairwise distance relationships:

$$\mathcal{L}_{RKD} = \frac{1}{b^2} \sum_{i \neq j} \left( \frac{d_{ij}^S}{\|D^S\|_2} - \frac{d_{ij}^T}{\|D^T\|_2} \right)^2$$

where $d_{ij} = \|f(x_i) - f(x_j)\|_2$ and $D$ is the distance matrix. RKD preserves structural relationships without direct feature alignment.

## 4. Experimental Setup

### 4.1 Dataset and Preprocessing

We conduct all experiments on the **CIFAR-100** dataset [19], which consists of 60,000 32×32 color images across 100 classes (50,000 training and 10,000 test images). Standard data augmentation is applied: random 32×32 crops with 4-pixel padding, random horizontal flips, and color jitter. Input images are normalized using the dataset mean (0.5071, 0.4867, 0.4408) and standard deviation (0.2675, 0.2565, 0.2761).

### 4.2 Network Architectures

**Teacher network:** ResNet-34 (21.3M parameters), trained from scratch on CIFAR-100 for 60 epochs (best accuracy: 76.75%).

**Student network:** ResNet-18 (11.2M parameters), trained from random initialization.

The teacher-student architecture gap (ResNet-34 → ResNet-18) represents a moderate capacity difference suitable for evaluating both symmetric and asymmetric methods.

### 4.3 Training Protocol

| Hyperparameter | Value |
|:---|---:|
| Optimizer | SGD with Nesterov momentum |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Batch size | 128 |
| Epochs | 60 |
| Initial learning rate | 0.05 |
| LR schedule | Cosine annealing [20] |
| Warmup epochs | 5 |

### 4.4 KD Method Hyperparameters

| Method | Category | Hyperparameters |
|:---|---:|:---|
| KD [7] | Symmetric | $\tau=4.0$, $\alpha=0.5$ |
| FitNet [8] | Symmetric | $\beta=1000$ |
| AT [9] | Symmetric | $\beta=500$ |
| SP [10] | Asymmetric | $\beta=3000$ |
| CC [11] | Asymmetric | $\beta=0.02$, $\gamma=0.4$ |
| RKD [12] | Asymmetric | $\beta=1.0$ |

### 4.5 Evaluation Protocol

All experiments are conducted on a single NVIDIA RTX 3090 GPU. Each method is trained for exactly 60 epochs with matched random seeds, data splits, and augmentation pipeline. We report the **best test accuracy** across all epochs and full learning curves.

**Limitation:** Results are based on a single training run per method due to computational constraints. While this is standard practice in exploratory comparative studies [18], we acknowledge that multi-run experiments with statistical significance testing would strengthen the conclusions. The reported differences should be interpreted with appropriate caution.
