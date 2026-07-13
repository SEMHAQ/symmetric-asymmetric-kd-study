# Symmetric vs. Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

**Author Name**  
*Affiliation*  
*Correspondence: email@example.com*

---

## Abstract

Knowledge distillation (KD) is a widely adopted technique for compressing deep neural networks. Despite the proliferation of KD methods, a systematic understanding of their underlying mechanisms remains elusive. In this paper, we propose a perspective that categorizes KD methods along a symmetry spectrum: **symmetric** methods directly align teacher and student representations through one-to-one feature matching, while **asymmetric** methods preserve relational structures among samples without requiring direct feature correspondence. Through extensive experiments on CIFAR-100 with ResNet-34 as teacher and ResNet-18 as student, we compare six representative KD methods (KD, FitNet, AT, SP, CC, RKD). Our results reveal three key findings: (1) both symmetric and asymmetric methods can effectively improve student performance; (2) symmetric methods converge faster while asymmetric methods achieve competitive or higher peak accuracy; (3) the boundary between symmetric and asymmetric methods is a spectrum rather than a dichotomy, with several methods exhibiting hybrid characteristics. This work provides practitioners with a complementary conceptual framework for understanding and selecting knowledge distillation strategies.

**Keywords:** knowledge distillation; symmetric learning; asymmetric learning; model compression; image classification; representation learning
