# Perspective Review Report

**Reviewer:** Peer Reviewer 3 (Cross-Disciplinary / Geometric Deep Learning)
**Paper:** Symmetric versus Asymmetric Knowledge Distillation: A Comparative Study for Image Classification

---

## Cross-Disciplinary Contribution: 5/10

### Strengths
- The connection to SimSiam/BYOL's symmetric vs. asymmetric architectures is well-drawn
- The paper is clearly written and accessible
- The "symmetry spectrum" concept usefully complicates a binary classification

### Weaknesses

**W1. Superficial symmetry analysis (CRITICAL for Symmetry journal).** The paper uses "symmetric" as a label without engaging with the mathematical concept. For a journal titled *Symmetry*, the paper should address at minimum:
- What symmetry group (if any) is preserved or broken by each class of methods?
- Can the distinction be formalized in terms of equivariance (symmetric) vs. invariance (asymmetric)?
- How does this relate to known concepts like gauge symmetry or group-equivariant representations?
Without this depth, the paper risks appearing to use "symmetry" opportunistically.

**W2. Self-supervised learning connection underdeveloped (MAJOR).** The paper mentions SimSiam and BYOL but does not leverage insights from that literature. SimSiam demonstrates that asymmetric predictor networks prevent representational collapse. Could asymmetric KD similarly prevent student overfitting to teacher noise? This is a testable hypothesis the paper does not explore.

**W3. Practical guidelines lack quantification (MINOR).** The observations in Section 6 are qualitative. Quantifying trade-offs (e.g., "symmetric KD trains 30% faster but converges to 0.5% lower accuracy") would significantly increase practical value.

**W4. Broader impact beyond image classification (MINOR).** Could this taxonomy apply to other domains where KD is used (NLP, speech, graph neural networks)? A brief discussion would broaden relevance.

## Recommendation: Major Revision

The paper requires a substantive engagement with symmetry as a mathematical concept to be suitable for Symmetry journal. I recommend adding a subsection that formalizes the symmetric/asymmetric distinction in terms of equivariance and invariance properties.
