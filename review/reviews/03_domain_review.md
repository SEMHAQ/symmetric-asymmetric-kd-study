# Domain Review Report

**Reviewer:** Peer Reviewer 2 (Knowledge Distillation Specialist)
**Expertise:** Model compression, knowledge distillation, representation learning
**Paper:** Symmetric vs. Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Overall Assessment

The paper addresses an interesting gap in the KD literature—the lack of a systematic taxonomy for understanding KD methods. The proposed symmetric/asymmetric classification is intuitive and potentially useful.

---

## Domain Contribution

**Score: 6/10**

### Strengths
- The taxonomy fills a genuine gap in how practitioners think about KD methods
- The comparison is fair and well-controlled
- The learning curve analysis provides useful insights beyond simple accuracy rankings

### Weaknesses

1. **Missing key methods (MAJOR).** Several influential KD methods are omitted from the comparison:
   - Relational Knowledge Distillation (RKD) [Park et al., 2019]—this is explicitly relational (asymmetric) and would be a natural inclusion
   - Contrastive Representation Distillation (CRD) [Tian et al., 2019]—a popular method
   - Knowledge Review [Chen et al., 2021]—uses multi-level feature alignment
   - The omission weakens the claim of being "comprehensive"

2. **Theoretical depth of taxonomy (MAJOR).** The distinction between symmetric and asymmetric is defined operationally (what the loss function looks like) rather than theoretically. The paper would benefit from:
   - A theoretical analysis of when each type should work better
   - Information-theoretic interpretation (e.g., symmetric methods maximize mutual information between T and S representations; asymmetric methods preserve structural entropy)

3. **Literature coverage gaps (MINOR).** The related work section on KD is adequate but brief. Consider adding:
   - Born-Again Networks [Furlanello et al., 2018]
   - Distillation with No Teacher [Zhang et al., 2019]
   - Self-distillation literature

4. **Claims about "first" taxonomy (MINOR).** While the explicit symmetric/asymmetric framing may be novel, prior works have discussed similar distinctions (e.g., "instance-level vs. structure-level" distillation). Please qualify the novelty claim more carefully.

---

## Recommendation

**Minor to Major Revision**

Include additional KD methods (at minimum RKD and CRD) to strengthen the comprehensiveness of the comparison. Add theoretical depth to the taxonomy discussion.
