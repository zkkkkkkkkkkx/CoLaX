
# CoLaX: Context-Aware Local Explanations for Time Series Classification

## 🔍 Overview

**CoLaX (Context-aware Local eXplanations)** is a **post-hoc explanation framework for time series classification**. Many existing explanations identify salient time steps or local motifs without explicitly considering the broader temporal context. Yet the same local pattern can have different predictive relevance in different sequence-level regimes.

CoLaX addresses this **local–global ambiguity** by learning **local temporal prototypes** alongside **global context representations**. An **input-dependent gating mechanism** conditions the relevance of local patterns on the global sequence. The resulting representations guide structured, segment-level perturbations of a separately trained, frozen black-box classifier, and **Shapley-based attribution** produces a temporal saliency map.

💡 **Key idea:** A local pattern should not be interpreted independently of the temporal context in which it occurs.

## ✨ Highlights

- 🧩 **Local Pattern Discovery:** Learn a dictionary of discriminative temporal prototypes and detect their occurrences within the input sequence.
- 🌐 **Global Context Modeling:** Encode long-range temporal structure with a Transformer-based encoder and a learnable context memory bank.
- ⚙️ **Context-Conditioned Modulation:** Use sample-specific global information to adjust the relevance of local prototypes rather than treating each motif as universally important.
- 🎯 **Post-hoc Black-box Attribution:** Build context-aware, shapelet-aligned segments and quantify their contributions using structured perturbations and Shapley values, without modifying the classifier being explained.
- 📊 **Interpretability Evaluation:** Assess correspondence to reference saliency on annotated benchmarks and use perturbation-based faithfulness tests when reference explanations are unavailable.
