# Literature survey: N/K scaling law for zero-shot voice copying

Survey date: 16 November 2026. Method: web searches with generic queries, plus reading the full text
of five PDFs downloaded from arXiv (indicated below).

**Verification legend** (applies to all entries):

- **[READ]** — I opened the PDF and read the cited table/passage; the numbers were copied from the text.
- **[ABSTRACT]** — I only saw the abstract or the snippet returned by the search; formulas and numbers come
  from that summary and were **not** checked against the full text.
- **[MEMORY]** — a claim I know from memory that the search did **not** confirm; check before citing.
- **[MY CALCULATION]** — a calculation I made on third-party numbers; not a claim by the authors.

---

## (a) Works by topic

### 1. Multi-speaker TTS / voice cloning: similarity vs. number of training speakers

Overall search result: **I found no work that states an explicit scaling law
(functional form + exponent) for zero-shot speaker similarity as a function of the number of speakers N**,
nor one that varies N and K (sentences per speaker) factorially. What exists are qualitative ablations,
almost always confounded with a change of corpus or with total hours.

| Work | Link | What it shows | Status |
|---|---|---|---|
| Transfer learning from speaker verification to multi-speaker TTS (Tacotron 2 + d-vector, NeurIPS 2018) | https://arxiv.org/abs/1806.04558 | The work closest to ours in architecture. States: "zero-shot transfer requires training on thousands of speakers". Table 5 varies the training set of the **speaker encoder** (synthesizer fixed at 1.2K speakers): 1.2K speakers → SV-EER 16.60% / 15.32%, similarity MOS 2.23 / 2.27; 2.4K → 11.95%, 2.43; 8.4K → 10.14%, 2.54; 18K (internal) → 5.08%, 3.03. No curve fit, no exponent; the points mix corpora and embedding dimensions (64 vs 256). | [READ] |
| same — implicit exponent | — | From 1.2K to 8.4K speakers (7x) the EER drops from 15.3–16.6% to 10.1%: apparent exponent ≈ 0.21–0.25. Including the 18K point (different internal corpus) the exponent rises to ≈ 0.41. **It is a suggestive coincidence with our 0.26, but completely confounded** (corpus, architecture, and it is the encoder's N, not the synthesizer's). | [MY CALCULATION] |
| SC-GlowTTS | https://arxiv.org/abs/2104.05557 | Zero-shot with only 11 training speakers (VCTK), claiming state-of-the-art similarity for new speakers; points to the seen/unseen gap as an open problem. Does not vary N. | [ABSTRACT] |
| YourTTS | https://arxiv.org/abs/2112.02418 | Argues qualitatively that limiting the number/variety of speakers hurts generalization to unseen voices. No curve. | [ABSTRACT] |
| Glow-WaveGAN 2 | https://arxiv.org/abs/2207.01832 | Seen/unseen gap larger on VCTK than on LibriTTS, attributed to the smaller number of speakers; a pre-trained encoder reduces the gap. Comparison between corpora, not a sweep. | [ABSTRACT] |
| Common Voice for multi-speaker TTS (GlowTTS) | https://arxiv.org/abs/2210.06370 | Quality-filtered subsets with different numbers of speakers (e.g. 633 → 537 → ...). Concludes that "more training speakers are key for modelling speaker variability", but N is confounded with the quality threshold; similarity for unseen speakers is non-monotonic. | [READ partially: table 1 and discussion] |
| Pre-training strategies for end-to-end multi-speaker TTS | https://arxiv.org/abs/2011.04839 | Including more speakers (LibriTTS) in initial training reduces overfitting to the seen speakers. Qualitative. | [ABSTRACT] |
| Zero-shot TTS with state-of-the-art neural speaker embeddings | https://arxiv.org/abs/1910.10838 | Compares embedding types (x-vector, LDE) for unseen speakers; does not vary N. | [ABSTRACT] |
| Cross-lingual multi-speaker TTS for unseen speakers | https://arxiv.org/abs/1911.11601 | Has a table of the effect of increasing training speakers on verification EER; **I did not see the numbers**. | [ABSTRACT] |
| ECAPA vs x-vector in YourTTS (2025) | https://arxiv.org/abs/2506.20190 | States that the encoder and the number of distinct speakers are determinants of similarity; no law. | [ABSTRACT] |
| VALL-E | https://arxiv.org/abs/2301.02111 | Scales to 60K hours / ~7K speakers and shows a large gain in zero-shot similarity; acknowledges degradation for unseen speakers in earlier systems. Does not isolate N. | [ABSTRACT] |
| VALL-E 2 | https://arxiv.org/abs/2406.05370 | Trained on Libriheavy (50K h, ~7000 speakers — checked in the text). The search summarizes that SIM improves with more data; **I did not locate the data-size ablation in the text**. | [READ partially] |
| MegaTTS 3 | https://arxiv.org/abs/2502.18924 | Clearest data curve I found: SIM-O 0.52 (2K h) → 0.65 (200K h) → 0.66 (600K h). Varies **hours**, not speakers; saturates at the top. | [ABSTRACT] |
| CLaM-TTS, DiTTo-TTS, Mega-TTS, WavTTS | https://arxiv.org/abs/2404.02781 · https://arxiv.org/abs/2406.11427 · https://arxiv.org/abs/2306.03509 · https://arxiv.org/abs/2606.03455 | All report that more data improves similarity; none separates number of speakers from total hours or fits a functional form. | [ABSTRACT] |
| MimicLM (2026) | https://arxiv.org/abs/2604.11552 | The search returned the sentence that similarity gains "follow typical scaling-law patterns with diminishing returns" without saturating. The abstract does not say this; **I did not check it in the text**. It is about data volume, not N. | [ABSTRACT, not confirmed] |
| Hierarchical generative modeling for controllable speech synthesis (GMVAE-Tacotron, ICLR 2019) | https://arxiv.org/abs/1810.07217 | Same family as our model (Tacotron 2 + VAE latents). Shows one-shot inference of speaker attributes for unseen speakers. Does not study the dependence on N or K. | [ABSTRACT] |
| Speaker Generation (TacoSpawn) | https://arxiv.org/abs/2111.05095 | Fits a parametric prior over speaker embeddings and samples new speakers — evidence that the "speaker space" can be modeled as a low-complexity distribution. | [ABSTRACT] |

**Scaling laws in speech outside TTS** (for contrast): power laws of WER vs. data/parameters in
ASR rescoring (https://arxiv.org/abs/2306.15815), in OWLS multilingual foundation models
(https://arxiv.org/abs/2502.10373) and in LLM-based ASR vs. FLOPs (https://arxiv.org/abs/2508.04096)
— all [ABSTRACT]. In other words, "scaling laws in speech" exist for ASR/hours, not for N speakers in TTS.

### 2. Speaker verification / embeddings: generalization vs. number of speakers

Here too **I did not find a fitted power law of EER vs. number of speakers**.

| Work | Link | What it shows | Status |
|---|---|---|---|
| Training speaker recognition with limited data ("tiny" VoxCeleb2 subsets) | https://arxiv.org/abs/2203.14688 | Fixed budget of ~50K utterances, three designs: few speakers (100 spk., ~51 sessions each), many speakers/few sessions (5994 spk., 1 session), many speakers/many sessions (5994 spk., ~8 sessions, 1 utterance per session). EER (dev, vox1-o) X-vector / ECAPA / wav2vec2: 12.9 / 12.2 / 7.5% · 21.7 / 16.0 / 15.7% · 9.8 / 6.0 / 3.6%; full VoxCeleb2: 6.3 / 2.9 / 2.4%. Authors' conclusion: many speakers **without** session variability is the worst case; the initial hypothesis "maximizing speakers is paramount" does not hold on its own. It is the most direct analogue of our N×K trade-off, but with 3 points and no functional form. | [READ] |
| VoxBlink2 (100K+ speakers) | https://arxiv.org/abs/2407.11510 | Random subsets of 5K, 10K, 30K and 110K speakers. In mixed fine-tuning, EER 0.527 / 0.505 / 0.505 / 0.674% — "does not improve consistently"; in pre-training, Figure 3 shows that more speakers helps (I did not read the values from the figure). No fit. | [READ partially] |
| Empirical study of GE2E | https://arxiv.org/abs/2011.04896 | Effect of the number of enrollment utterances per speaker: decisive for low values, saturates afterwards (compatible with a 1/K-type term). | [ABSTRACT] |
| Base dataset design in few-shot image learning (ECCV 2020) | https://arxiv.org/abs/2007.08872 | Analogue outside speech: with a fixed annotation budget, the classes × images-per-class trade-off has a large effect; the optimum uses many classes and few images per class. Empirical, no law. | [ABSTRACT] |
| Generating novel classes for metric learning | https://arxiv.org/abs/2201.01008 | With a fixed total number of samples, Recall@1 on unseen classes grows with the number of training classes. Empirical. | [ABSTRACT] |

### 3. Theory of neural scaling laws and exponents from intrinsic dimension

| Work | Link | What it shows | Status |
|---|---|---|---|
| A Neural Scaling Law from the Dimension of the Data Manifold | https://arxiv.org/abs/2004.10802 | Loss ∝ (no. of parameters)^(-α) with **α ≈ 4/d**, d = intrinsic dimension of the data manifold; tested on teacher/student, CNNs and GPT (for GPT: 4/α ≈ 53 vs. d ≥ 90, i.e., only the inequality holds). Note: the law is in **parameters**, not in number of samples. | [ABSTRACT] |
| Explaining Neural Scaling Laws (PNAS 2024) | https://arxiv.org/abs/2102.06701 | Four regimes: variance-limited (universal exponent, coming from the existence of a well-behaved limit) and resolution-limited (exponent depends on the spectrum of the kernel / of the data manifold), along the data and width axes; duality between the two exponents. In real networks the exponents are less sharp than in teacher-student. | [ABSTRACT] |
| Asymptotic learning curves of kernel methods | https://arxiv.org/abs/1905.10843 | Gaussian teacher-student: error ∝ n^(-β), β fixed by smoothness and dimension (β = (α_T − d)/d when the student is not limiting); via kernel-PCA estimates β ≈ 0.36 (MNIST) and ≈ 0.07 (CIFAR10), in agreement with observation. Mentions the baseline β = 1/d for Lipschitz functions. Exact refinement: https://arxiv.org/abs/2006.09754. | [ABSTRACT] |
| Spectrum-dependent learning curves in kernel regression | https://arxiv.org/abs/2002.02561 | Analytical expressions for the generalization error as a function of the kernel eigenspectrum and the target alignment. | [ABSTRACT] |
| Learning Curve Theory | https://arxiv.org/abs/2102.04074 | Toy model that produces n^(-β) for any β > 0; the exponent depends on the data distribution (Zipf-like tail), it is not universal. | [ABSTRACT] |
| The Quantization Model of Neural Scaling (NeurIPS 2023) | https://arxiv.org/abs/2303.13506 | A power law of the loss emerges if skill "quanta" have power-law frequencies and are learned in order of frequency. Alternative to the dimension-based explanation. | [ABSTRACT] |
| Deep Learning Scaling is Predictable, Empirically | https://arxiv.org/abs/1712.00409 | Power-law learning curves in translation, LM, images and **speech**; exponents measured, not derived; model improvements shift the curve without changing the exponent. | [ABSTRACT] |
| A Constructive Prediction of the Generalization Error Across Scales (ICLR 2020) | https://arxiv.org/abs/1909.12673 | Joint functional form in model size and data size that extrapolates from small to large scales. Methodological standard for "predict and then test". | [ABSTRACT] |
| DNNs adapt to intrinsic dimension | https://arxiv.org/abs/1907.02177 | Rate O(n^(-2β/(2β+d))) with d = intrinsic (Minkowski) dimension of the support, without requiring manifold structure. | [ABSTRACT] |
| Nonparametric regression on manifolds with deep ReLU networks | https://arxiv.org/abs/1908.01842 | Squared error n^(-2(s+α)/(2(s+α)+d)) log³n for a Hölder target on a d-dimensional Riemannian manifold. Extension without the manifold assumption: https://arxiv.org/abs/2306.14859. | [ABSTRACT] |
| k-NN adapts to local intrinsic dimension (NeurIPS 2011) | https://arxiv.org/abs/1110.4300 | Near-minimax rate Õ(n^(-2/(2+d))) with local and unknown d. | [ABSTRACT] |
| Convergence rates of empirical measures in Wasserstein distance (Bernoulli 2019) | https://arxiv.org/abs/1707.00087 | E[W_p(μ̂_n, μ)] ~ n^(-1/d), with d an **intrinsic** dimension of the measure; multi-scale behavior (different rates in different ranges of n). | [ABSTRACT] |
| Generalization via embedding dimension and distributional convergence (ICML 2026) | https://arxiv.org/abs/2601.22756 | Bound on the generalization gap = (Wasserstein rate fixed by the intrinsic dimension of the embeddings) × (Lipschitz constant of the embedding→prediction map). **It is the argument closest to what we want to make**, but in i.i.d. classification, not in generalization to new groups. | [ABSTRACT] |
| Intrinsic dimension of images and its impact on learning (ICLR 2021) | https://arxiv.org/abs/2104.08894 | Measures the intrinsic dimension of image datasets (low) and shows, controlling it with GANs, that it — and not the ambient dimension — governs sample complexity. An experimental model for us. | [ABSTRACT] |
| Minimax rate n^(-2p/(2p+d)) (nonparametric regression, 1982) | — | Classical result; **the search did not return the source**. | [MEMORY] |

### 4. Theory of generalization to new groups/tasks

| Work | Link | What it shows | Status |
|---|---|---|---|
| A model of inductive bias learning (2000) | cited in https://arxiv.org/abs/1505.06279 | Formalizes a "task environment": tasks drawn i.i.d. from a distribution; error analysis via covering numbers in number of tasks and samples per task. It is the exact conceptual framework (speaker = task). | [ABSTRACT, via citation] |
| The Benefit of Multitask Representation Learning (JMLR 2016) | https://arxiv.org/abs/1505.06279 | Uniform convergence bounds for learning-to-learn of order **O(1/√T) + O(1/√m)** (T tasks, m samples per task); the 1/√T term cannot be improved in this framework. Corresponds to our form a + A/√N + C/√K. The exact form of the theorem **was not read**; the order 1/√T + 1/√m came from the description in the paper of the next item. | [ABSTRACT] |
| On the Theory of Transfer Learning: The Importance of Task Diversity (NeurIPS 2020) | https://arxiv.org/abs/2006.11650 | With "diverse" tasks: the sample complexity for learning the representation scales as C(H) + t·C(F); a new task costs only C(F); the rate decays with **all** samples (n·t). Introduces a diversity parameter ν. | [ABSTRACT] |
| Few-Shot Learning via Learning the Representation, Provably (ICLR 2021) | https://arxiv.org/abs/2002.09434 | Linear representation of dimension k in ambient dimension d: risk on the target task **O(dk/(n₁T) + k/n₂)**; bypasses the Ω(1/T) barrier of the i.i.d.-task setting thanks to a coverage/diversity assumption. The first term is exactly our B/(NK). | [ABSTRACT] |
| Shared linear representations, with lower bounds | https://arxiv.org/abs/2002.11684 | Rates in T and samples per task with nearly optimal lower bounds. Details not read. | [ABSTRACT] |
| A Distribution-Dependent Analysis of Meta-Learning (ICML 2021) | https://arxiv.org/abs/2011.00344 | Gaussian linear regression with a Gaussian task distribution (= random-effects model): problem-dependent lower bounds on the transfer risk and an estimator that attains them up to a constant. **It is the theorem closest to the form a + A/N + B/(NK)**; the exact expression was not read. | [ABSTRACT] |
| Theoretical bounds on estimation error for meta-learning | https://arxiv.org/abs/2010.07140 | Two-level Bayesian model for linear regression; minimax lower bounds and an upper bound (MAP) for a new task. | [ABSTRACT] |
| Generalizing from Several Related Classification Tasks to a New Unlabeled Sample (NeurIPS 2011) | https://proceedings.neurips.cc/paper/2011/hash/b571ecea16a9824023ee1af16897a582-Abstract.html | Formalizes domain generalization: N training domains drawn from a meta-distribution; error analysis and universal consistency via kernels on distributions. The rate in N did not appear in the snippets. | [ABSTRACT] |
| Domain Generalization by Marginal Transfer Learning (JMLR 2021) | https://arxiv.org/abs/1711.07910 | Extended version; bound in N domains and n samples per domain (Rademacher). Exact form not seen. | [ABSTRACT] |
| How Many Domains Suffice for Domain Generalization? (NeurIPS 2025) | https://arxiv.org/abs/2506.16704 | Defines "domain sample complexity" and a domain shattering dimension; matching upper and lower bounds up to polylog. Formally answers "how many domains (speakers) are enough", but in PAC classification. | [ABSTRACT] |
| Theoretical analysis of unsupervised contrastive learning (ICML 2019) | https://arxiv.org/abs/1902.09229 | "Latent classes" framework; guarantees for downstream tasks formed by the **same** latent classes — does not cover truly new classes. | [ABSTRACT] |
| Two-stage cluster sampling | (sampling textbooks; the search returned only tutorial pages) | The variance of the mean estimator with N groups and K units per group decomposes into a between-group component (∝ 1/N) and a within-group component (∝ 1/(NK)); design effect 1 + (K−1)ρ. It is the elementary statistical origin of the form a + A/N + B/(NK). The explicit formula did **not** appear in a primary source in the search. | [MEMORY + tutorial pages] |

### 5. Information-theoretic measures

| Work | Link | What it shows | Status |
|---|---|---|---|
| Fixing a Broken ELBO (ICML 2018) | https://arxiv.org/abs/1711.00464 | Rate-distortion plane for VAEs: variational lower and upper bounds on the mutual information I(x;z); models with the same ELBO can have very different rates; powerful decoders ignore the latent. Our measure "bits per utterance = KL" is the **rate R** of this framework (upper bound on I). | [ABSTRACT] |
| Capacity of the variational embedding in expressive TTS (Capacitron) | https://arxiv.org/abs/1906.03402 | In Tacotron with a variational reference encoder: defines capacity = mutual information embedding↔data, upper-bounded by the KL; capacity is controlled by constraining the KL (in bits/nats); more capacity → more transfer precision, less generality. **Direct precursor of our 65–73-bit measure**; does not decompose capacity into "speaker" bits nor relate it to N. | [ABSTRACT] |
| Uniqueness of the i-vector representation of voice | https://arxiv.org/abs/2008.11985 | Estimates the biometric information of voice: 43–52 (or 43–70) bits in a constrained condition with 5 s samples; drops to **13–20 bits** in less constrained conditions; doubling the duration adds ~20 bits. Earlier estimates cited range from 14 to 120+ bits. Our 10–20 identity bits fall in the "unconstrained" range. | [ABSTRACT] |
| Privacy disclosure in speech measured in bits (2025) | https://arxiv.org/abs/2508.05250 | Measures in bits how much each type of feature reveals about identity; disclosures of independent features are additive; speaker recognition embeddings are the most informative. I did not see the numerical values. | [ABSTRACT] |
| Interpreting the dimensions of speaker embedding space (2025) | https://warwick.ac.uk/fac/soc/al/research/baap2026/ladefoged/interpreting-the-dimensions-of-speaker-embedding-space.pdf | Neighborhood preservation (UMAP) saturates after ~6 dimensions; 9 interpretable acoustic parameters explain the same as 7 principal components (>50% of the variance). An indication — not a formal estimate — of low effective dimension (~6–10). | [ABSTRACT] |
| M-Vec: "Matryoshka" speaker embeddings | https://arxiv.org/abs/2409.15782 | Speaker embeddings with flexible dimension; according to the proposal, dimensions much smaller than 192–512 retain much of the performance (I did not see the numbers). | [ABSTRACT] |
| Eigenvoices (rapid speaker adaptation, ~2000) | https://eurecom.fr/publication/200/download/mm-nguypa-990301b.pdf | Speakers live in a low-dimensional linear subspace; adapting = estimating a few coefficients. Classical precedent for a "low-dimensional speaker space". | [ABSTRACT] |
| How much do language models memorize? (2025) | https://arxiv.org/abs/2505.24832 | Memorization capacity of GPT models ≈ 3.6 bits per parameter (3.5–4); when capacity fills up, unintended memorization drops and generalization begins; derives scaling laws for membership inference. A model of "capacity predicted and then tested". | [ABSTRACT] |

**Not found**: a formal estimate (TwoNN, MLE etc.) of the intrinsic dimension of x-vectors/d-vectors;
nor a measure of "speaker identity bits" inside a TTS style latent as a function of N.

### 6. "Minimal model" / capacity saturation predicted and tested

| Work | Link | What it shows | Status |
|---|---|---|---|
| Don't Blame the ELBO! (NeurIPS 2019) | https://arxiv.org/abs/1911.02469 | A linear VAE recovers pPCA; the number of active latent dimensions is determined by the observation noise vs. the eigenvalues; the linear analysis predicts the behavior of nonlinear VAEs. Basis for predicting the "minimum latent size". | [ABSTRACT] |
| Dataset-size dependence of the rate-distortion curve and of the collapse threshold in a linear VAE | https://arxiv.org/abs/2309.07663 | Closed form relating β, dataset size, posterior collapse and the rate-distortion curve in a linear VAE in the high-dimensional limit. **The only work found that links the VAE's R(D) to dataset size** — but i.i.d., without group structure. | [ABSTRACT] |
| 2026 preprint on collapse thresholds in β-VAE | https://arxiv.org/abs/2605.22691 | In the linear Gaussian case, collapse thresholds, distortion reductions and normalized PCA eigenvalues coincide; warns that "β gives N active dimensions" is not intrinsic without normalization. Seen only via an aggregator. | [ABSTRACT, not peer-reviewed] |
| Capacitron | https://arxiv.org/abs/1906.03402 | (see item 5) Varies the target capacity and observes the precision/generality trade-off — an empirical test of capacity in TTS, without a theoretical prediction of the minimum. | [ABSTRACT] |
| How much do language models memorize? | https://arxiv.org/abs/2505.24832 | (see item 5) Prediction of capacity saturation confirmed across hundreds of transformers. | [ABSTRACT] |

I did not find *information bottleneck* results (phase transitions / minimal sufficient representation)
in the search; if used, it needs a dedicated search.

---

## (b) What already exists and what would be new

### What already exists (with reasonable confidence)

1. **Qualitative fact**: more training speakers improve zero-shot copying; "thousands of speakers are
   needed" (2018). Well established and repeated in almost every zero-shot TTS paper.
2. **N×K trade-off under a fixed budget**, in speaker verification (tiny VoxCeleb2 subsets) and in
   few-shot image learning — empirical, with 3–4 points, no functional form.
3. **The forms a + A/N + B/(NK) and a + A/√N + C/√K** are, respectively, the random-effects /
   two-stage sampling form and the form of the uniform convergence bounds of learning-to-learn.
   As *forms*, they are not new.
4. **Exponent from intrinsic dimension** (≈4/d in parameters; n^(-2s/(2s+d)) and n^(-1/d) in samples)
   is an established idea for i.i.d. learning, and in 2026 there is already an "embedding dimension × Wasserstein" bound.
5. **KL of the style latent as capacity in bits** in variational Tacotron (Capacitron), and **identity
   bits in voice** (13–70 bits, i-vectors).

### What seems new (as far as the search reached)

- **A joint quantitative law in N and K for zero-shot voice copying**, with a clean separation
  "seen depends on K, unseen depends on N", in a factorial design. I found nothing equivalent in TTS
  or in speaker verification.
- **Tying the exponent in N to the intrinsic dimension of the speaker space** (treating speakers, not
  utterances, as the samples of the manifold). Existing theory treats i.i.d. samples; the transposition
  to "samples = groups" with K noisy observations per group did not appear.
- **Decomposition of the latent's KL into identity bits vs. the rest, growing as log N**, and a
  **minimum latent size predicted and tested**. I found no analogue.

### Closest prior work

- Empirical: https://arxiv.org/abs/1806.04558 (Table 5) and https://arxiv.org/abs/2203.14688.
- Theoretical: https://arxiv.org/abs/2011.00344 and https://arxiv.org/abs/2002.09434 (form in N and NK);
  https://arxiv.org/abs/2004.10802, https://arxiv.org/abs/2102.06701 and https://arxiv.org/abs/2601.22756
  (exponent from dimension).
- Information-theoretic: https://arxiv.org/abs/1906.03402 and https://arxiv.org/abs/2008.11985.

### Honest caveats about novelty

1. **Absence of evidence is not evidence of absence.** There were ~35 generic searches; Interspeech/ICASSP
   and theses are poorly indexed. A speech reviewer may know an N ablation I did not find. Before
   claiming "first", run a dedicated search in the ISCA Archive and IEEE Xplore.
2. **The two candidate forms contradict the observed exponent.** The hierarchical form predicts
   1/N decay; the uniform convergence bounds, 1/√N. The empirical data show N^(-0.26) with no floor. Such a
   small exponent is a signature of a **nonparametric / resolution-limited** regime, not of finite-dimensional
   random effects. This is at once the most interesting and the most fragile part: with 16 runs,
   the two forms fit "equally well" precisely because the floor *a* absorbs the difference.
3. **Three values of N (8, 32, 128) do not distinguish a power law from logarithmic decay** nor from
   a power law with a floor. 43→30→21% gives a ratio of 0.70 at both steps (clean), but that is only two intervals.
4. **The exponent does not identify the dimension without an additional assumption.** [MY CALCULATION] 0.26 corresponds to
   d ≈ 4 if the rate is N^(-1/d); d ≈ 6 if it is N^(-2/(2+d)); d ≈ 8 if N^(-2/d); d ≈ 15 if N^(-4/d).
   The prediction is falsifiable only if d is measured **independently** (e.g., TwoNN/MLE on the
   speaker encoder's embeddings) and the smoothness is fixed a priori.
5. **The metric is a threshold** (fraction of samples more similar to the reference than to the default voice),
   not a smooth loss; regression error rates do not automatically transfer to a tail
   fraction. In addition, the metric depends on a specific speaker encoder.
6. **Small scale and old model.** Tacotron 2 + VAE with N ≤ 128 is very far from the regime of current
   systems (thousands of speakers, tens of thousands of hours), which saturate (SIM-O 0.65→0.66
   from 200K to 600K hours). "No visible floor" up to N=128 says nothing about N=10⁴.

**Verdict**: the *factorial empirical N×K law* and the *exponent ↔ speaker-space dimension link* are,
by all indications, new as a quantitative statement; the *theoretical ingredients* are all known.
The defensible contribution is the transposition and the test, not a new theorem — unless a
rate is proven for the nonparametric hierarchical setting (see (c), gap).

---

## (c) Reusable theorems and required assumptions

| Result | Term it provides | Assumptions we would have to make | Risk |
|---|---|---|---|
| Two-stage variance decomposition / random effects | A/N + B/(NK) | Speakers i.i.d. from a population; utterances i.i.d. given the speaker; linear/finite-dimensional estimator | Predicts exponent 1 in N — contradicts 0.26. Useful for the "seen" regime (C/K) and as a variance term. |
| Meta-learning with a Gaussian task distribution (https://arxiv.org/abs/2011.00344) | Lower and upper bounds on the risk on a new task | Linear regression, fixed design, Gaussian tasks | Linear; gives optimality of the hierarchical form, not the small exponent. |
| Shared representation (https://arxiv.org/abs/2002.09434, https://arxiv.org/abs/2006.11650) | dk/(NK) + k/n₂ | A representation of dimension k common to all tasks; **task diversity** (the N speakers cover the k directions) | In zero-shot n₂ is a single reference utterance; the term k/n₂ becomes the floor *a*. The diversity assumption is exactly what fails with small N. |
| Learning-to-learn via uniform convergence (https://arxiv.org/abs/1505.06279) | 1/√N + 1/√K | i.i.d. tasks; representation class of finite complexity | It is a worst-case upper bound, not a learning curve; do not use it as a "law". |
| Wasserstein rate with intrinsic dimension (https://arxiv.org/abs/1707.00087) | N^(-1/d) | Speaker distribution with intrinsic dimension d > 2p; compact support | Gives d ≈ 4 for 0.26 — plausible against the "~6 dimensions" of item 5, but requires a Lipschitz map from the embedding to the metric. |
| Embedding × Lipschitz bound (https://arxiv.org/abs/2601.22756) | (Wasserstein rate) × L | Lipschitz embedding→output map; i.i.d. samples | Ready-made template for the argument; we would have to replace "sample" with "speaker" and handle the noise of K utterances. |
| Adaptive nonparametric regression (https://arxiv.org/abs/1907.02177, https://arxiv.org/abs/1908.01842, https://arxiv.org/abs/1110.4300) | N^(-2s/(2s+d)) | Hölder-s speaker→voice map on a support of dimension d; the network attains the minimax rate | Two free parameters (s, d) for one exponent: not identifiable without measuring d. |
| Variance/resolution regimes (https://arxiv.org/abs/2102.06701) | Interpretation: N in the resolution regime, K in the variance regime | Well-behaved limit in K; power-law spectrum in N | Fits our finding qualitatively (K → C/K; N → small power). It is the most natural narrative framework. |
| VAE rate-distortion (https://arxiv.org/abs/1711.00464, https://arxiv.org/abs/1906.03402) | KL ≥ I(x;z); capacity in bits | None strong: KL is an **upper** bound on the mutual information | The "10–20 identity bits" require their own estimator (lower bound via a classifier); state which bound is used. |
| Information counting | Distinguishing N speakers requires ≥ log₂ N bits | Trivial (Fano) | Explains "grows as log N" **within training**; does not by itself explain generalization to unseen speakers. |
| Linear VAE / pPCA (https://arxiv.org/abs/1911.02469, https://arxiv.org/abs/2309.07663) | No. of active dimensions = no. of eigenvalues above the noise | Linear-Gaussian | Gives a testable prediction of the minimum latent if we measure the spectrum of the between-speaker variance. |

**Theoretical gap (where a new theorem could be)**: no result found covers the
*hierarchical and nonparametric* case — N groups sampled from a distribution of intrinsic dimension d, each
observed through K noisy samples, risk measured on a new group. The natural conjecture, combining the
ingredients above, is an error of the form

    floor + (resolution term in N, exponent fixed by d and smoothness) + (variance term ∝ 1/(NK) or 1/K)

This is **my conjecture**, not a result from the literature.

---

## (d) What a top-venue paper would need to contain

1. **More decades in N.** At least 5–6 values of N covering ≥ 2 decades (e.g., 8 → 1024+; LibriTTS has
   ~2.4K speakers), to distinguish power law, power law with a floor, and logarithm. The same for K.
2. **Seeds and uncertainty.** Several seeds per (N, K) cell, confidence intervals, and formal
   model comparison (AIC/BIC or cross-validation leaving cells out). With 16 runs the tie
   between the two forms is expected.
3. **Out-of-sample prediction.** Fit on small N and **predict** large N before running (the standard of
   "A Constructive Prediction..."). This is what turns curve fitting into a law.
4. **Independent measurement of the speaker-space dimension**, with ≥ 2 estimators (TwoNN, MLE,
   PCA spectrum) and ≥ 2 speaker encoders, and the exponent predicted **before** looking at the curve.
5. **Intervention on the dimension.** As in the image study with GANs: restrict the speaker population
   (a single gender, a narrow pitch range, or synthetic speakers sampled from a prior of controlled
   dimension) and show that the exponent changes as predicted. Without this, "exponent ↔ dimension" is a
   single-point correlation.
6. **Confound controls.** Separate N from total hours (fixed N·K vs. fixed K); balance gender/accent;
   fixed and sufficient test speakers (38 is few for tails); channel/recording (speaker identity
   is correlated with channel).
7. **Metric robustness.** Repeat with another speaker encoder, with continuous similarity (cosine, EER)
   in addition to the threshold fraction, and with a small subjective test. Show that the exponent is not a threshold artifact.
8. **Architectural generality.** At least a second model (e.g., a modern zero-shot TTS at reduced
   scale, or a voice conversion model) — otherwise the reviewer will say it is a property of Tacotron 2 + VAE.
   Ideally a second domain (e.g., speaker verification, or faces) to support "law".
9. **Rigorous information-theoretic part.** State the bounds (KL is an upper bound; identity bits via a classifier
   is a lower bound), show the rate-distortion curve varying β/latent dimension, and test the "minimum
   latent": predict the dimension/capacity below which copying degrades and verify the knee.
10. **Theory with explicit assumptions and a solvable case.** A minimal model (e.g., hierarchical
    linear-Gaussian, or teacher-student with speakers on a d-dimensional torus/sphere) in which the law is derived
    exactly and simulated, plus the bridge to the real network. State the convergence (in what sense, at
    what rate) and what would falsify the theory.
11. **Honest positioning.** Acknowledge that the forms 1/N, 1/(NK), 1/√N and the dimension-based exponents
    are known; the novelty is the hierarchical setting, the measurement in speech, and the test. Discuss the saturation
    observed in large-scale systems and why (or whether) the law stops holding there.

---

## Limitations of this survey

- Only five papers had passages of their full text read (1806.04558, 2203.14688, 2407.11510,
  2406.05370, 2210.06370); everything else is [ABSTRACT].
- Theorem formulas (items 3 and 4) come from summaries generated by the search and **must be checked**
  against the originals before entering any text.
- Classics not returned by the search (the 1982 minimax rate; "Scaling Laws for Neural Language Models";
  two-stage sampling formulas) are marked [MEMORY] or omitted.
- No search was done in the ISCA Archive / IEEE Xplore or in theses.
