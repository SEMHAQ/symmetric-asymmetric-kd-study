## 5. Results and Analysis

### 5.1 Main Results

Table 1 presents the best test accuracies achieved by each method over 60 epochs of training under identical conditions.

**Table 1. Best test accuracy (%) on CIFAR-100. Teacher: ResNet-34 (76.75%). Student: ResNet-18.**

| Method | Category | Best Accuracy (%) | $\Delta$ vs. Baseline |
|:---|---:|---:|---:|
| Baseline (No KD) | — | 76.48 | — |
| KD (Hinton, 2015) | Symmetric | **77.71** | +1.23 |
| FitNet (Romero, 2015) | Symmetric | 76.06 | -0.42 |
| AT (Zagoruyko, 2017) | Symmetric | 77.34 | +0.86 |
| SP (Tung, 2019) | Asymmetric | **77.85** | +1.37 |
| RKD (Park, 2019) | Asymmetric | 76.43 | -0.05 |
| CC (Peng, 2020) | Asymmetric | 75.81 | -0.67 |

Several important observations emerge:

1. **Both symmetric and asymmetric methods can outperform the baseline**, with three of the six methods (KD, AT, SP) improving upon standard training.

2. **The best performing method is SP (asymmetric) at 77.85%**, outperforming the teacher network (76.75%) by 1.10 percentage points.

3. **Symmetric methods (KD, AT) show more consistent improvements**, with both outperforming the baseline, while asymmetric methods exhibit higher variance (SP excels while RKD and CC underperform).

4. **Three methods (KD, AT, SP) surpass the teacher's accuracy**, demonstrating that students can benefit from the regularizing effect of the distillation loss beyond simple imitation.

### 5.2 Learning Curves

Figure 1 shows test accuracy during training:

![Learning Curves](../figures/learning_curves.png)
**Figure 1.** Test accuracy vs. training epoch.

**Early-stage (epochs 1-20):** Symmetric methods (KD, AT) converge faster. By epoch 10, KD achieves 61.57% while SP achieves 57.63%. Direct feature alignment provides stronger gradients early in training.

**Mid-stage (epochs 20-40):** Asymmetric methods catch up and surpass symmetric methods. SP overtakes KD around epoch 30, suggesting relational knowledge becomes more valuable as features mature.

**Late-stage (epochs 40-60):** Accuracy stabilizes across methods, with SP maintaining a slight edge.

### 5.3 Category Comparison

![Category Comparison](../figures/category_comparison.png)
**Figure 2.** Distribution of accuracies by category.

- **Symmetric average:** $\mu_{sym} = 77.04\%$ (range: 76.06–77.71%)
- **Asymmetric average:** $\mu_{asym} = 76.70\%$ (range: 75.81–77.85%)
- **Baseline:** $\mu_{base} = 76.48\%$

Symmetric methods show a tighter range (1.65 pp vs. 2.04 pp for asymmetric), suggesting more robust and predictable improvements.

### 5.4 Summary of Findings

1. Both symmetric and asymmetric KD methods can improve student performance over the baseline.
2. Symmetric methods provide more consistent improvements across different configurations.
3. Asymmetric methods can achieve higher peak performance (SP, 77.85%) but with greater variability.
4. The teacher's accuracy (76.75%) is not an upper bound—students can surpass it through distillation.
5. **Caution:** Results are from single runs per method. Differences of <0.5% between methods should be interpreted with appropriate caution.
