# Tacotron 2 VAE Attention Collapse Fixes

This document records the changes made to resolve the attention collapse (alignment failure) observed in `tts_ptbr_fonetico_v4` and other experiments after long training steps.

## Applied Fixes

### 1. Softmax Masking NaN Fix
**File:** `src/models/tacotron2_vae/model.py`
**Issue:** Padding masks used `-float("inf")`, which evaluated to `0.0 / 0.0 = NaN` during softmax for fully masked sequences (e.g., edge cases).
**Fix:** Changed `self.score_mask_value` to a large finite negative scalar (`-1e4`), which resolves cleanly to uniform weights without generating NaNs.

### 2. Location-Sensitive Tanh Saturation (Vanishing Gradients)
**File:** `src/models/tacotron2_vae/model.py`
**Issue:** Over long sequences (e.g., >500 frames), cumulative attention weights grew extremely large, causing the linear output of `LocationLayer` to saturate the `tanh` activation. This forced the activation derivative to 0, completely vanishing gradients ($< 10^{-10}$) and "freezing" the attention.
**Fix:** Added a `nn.LayerNorm(attention_dim)` and applied it to `processed_attention_weights` to scale the location features dynamically back to $O(1)$ and preserve healthy active gradients.

### 3. CoordConv Divide-by-Zero Fix
**File:** `src/models/tacotron2_vae/coord_conv.py`
**Issue:** Normalizing spatial coordinates with `(dim_y - 1)` caused a division-by-zero error whenever the mel spectrogram input had a sequence length of exactly 1 (common during single-frame generation or specific batching edge cases).
**Fix:** Implemented a safety floor: `max(dim_y - 1, 1)`.

### 4. VAE KL Loss Scaling
**File:** `src/training/training-tacotron2-vae/losses.py`
**Issue (Identified by subagents):** Standard implementations of KL divergence use a `sum` reduction. If the reconstruction loss uses a `mean` reduction, using `sum` for the KL loss causes a scale mismatch of ~100x, leading to Posterior Collapse where the KL loss dominates the objective.
**Status:** The current implementation in `losses.py` *already* correctly uses `kl_per_dim.mean(dim=0)` and `.mean()`. Therefore, no code changes were necessary here. It is documented as a confirmation of the correct design choice for maintaining proper ELBO scaling.
