*Historical note from the Colab phase. The organized, current research lives in [research/](../../research/README.md).*

# Why copying a new voice takes so many speakers: a theoretical treatment and a solution

This text organizes the results of the coverage experiment (README, section 9) into a simple model, checks the model against the 12 training runs of the grid, and derives a solution from it and tests it (the test came out negative, section 6). The argument is heuristic: the formulas are a sampling law fitted to the data, not a theorem.

## 1. What needs to be explained

The model reads a text in the voice of a reference audio clip. It is trained on 2,000 sentences from one voice (the "default voice", LJSpeech) plus N speakers with K sentences each. On new sentences, we measure the fraction of samples in which the reference voice was copied.

1. Copying of **trained** voices depends on K (sentences per speaker) and hardly depends on N.
2. Copying of **new** voices depends on N (number of speakers) and only weakly on K.
3. The farther the style code is from the default voice, the more the voice is copied and the more words come out wrong.
4. Four changes to the model did not reduce the need for data: taking the style from another sentence by the same speaker, a contrastive loss on the latent, a pre-trained speaker vector, and that vector with style dropout.

## 2. Model: training covers a cross, the test asks for the rectangle

The decoder is a function `f(c, s)`: `c` is the encoded text and `s` the style vector (which carries the voice). Generating well requires `f` to be correct at the requested pair `(c, s)`. The set of possible pairs is the product `C × S` (all texts × all voices), and training visits only part of it:

- one long row: many texts (2,000) in the default voice `s₀`;
- N short rows: K texts in each voice `sᵢ`.

The test asks for a new text `c*` in a voice `s`. The error at that point decomposes into two steps:

```
error(c*, s) = error(c*, s₀) + [error(c*, s) − error(c*, s₀)]
               \___________/    \__________________________/
               read new text      text × voice interaction:
               in default voice   what changes when the voice changes
```

The first term is small (WER of 4–6% with the style turned off), because the long row covers the texts well. The whole problem lies in the second term, and for it training only has the N·K samples of the short rows. This already explains fact 3: far from `s₀`, reading the text was trained on only K sentences per voice; the decoder either falls back to the default voice (voice not copied, text correct) or follows the voice and gets the text wrong. The style strength only picks the point between these two errors.

## 3. Two-level sampling law

The training data outside the default voice come from two-level sampling: N speakers are drawn and, within each one, K sentences. In any estimator of this kind (mixed models, cluster sampling) the error has three components, each falling with the count that feeds it:

- what is common to voices (how any voice alters speech) is learned from the N speakers: error ∝ 1/N;
- what is specific to a trained speaker is learned from that speaker's K sentences: error ∝ 1/K;
- the text × voice interaction is learned from all samples: error ∝ 1/(N·K).

A new voice lacks the second component (none of its sentences was seen), so the prediction for the fraction of samples in which the voice is **not** copied is:

```
new voices:      p = a + A/N + B/(N·K)
trained voices:  p = a' + C/K + A'/N
```

`a` and `a'` are what remains with infinite data of this kind.

**Fit to the 12 training runs** (step 2,500, full style; `scripts/coverage/fit_law.py`):

| Voices | Law | Parameters | Fit error (RMSE) |
|---|---|---|---|
| new | a + A/N + B/(NK) | a = 0.42; A = 1.03; B = 8.9 | 0.050 |
| new | a + A/N | a = 0.45; A = 1.05 | 0.068 |
| new | a + B/(NK) (total sentences only) | | 0.175 |
| new | a + C/K | | 0.184 |
| trained | a' + C/K + A'/N | a' = 0.08; C = 3.2; A' = 0.36 | 0.034 |
| trained | a' + C/K | a' = 0.15; C = 2.8 | 0.065 |
| trained | a' + A'/N | | 0.162 |

The deviation between two seeds of the same training run is 0.048. The two two-level laws fit the data down to the noise level; the single-factor alternatives (N only, K only, total N·K only) are 2 to 4 times higher. Caveat: these are 3 parameters for 7 configurations, with N ≤ 32; the fit does not distinguish 1/N from a slower decay.

## 4. What the theory explains and what it predicts

**Facts 1 and 2** are the law itself: K enters the trained voices through the term C/K, which does not exist for new voices; N dominates the new voices through the term A/N.

**The four negative results (fact 4).** All four changes act on how the voice is *represented* (what the latent carries, where the vector comes from). None of them adds (text, voice) pairs to the decoder's training, which is what A/N and B/(NK) count. The pre-trained speaker vector gives good coordinates for the voice space, but the decoder still sees only N points in it: with 8 speakers it memorizes (71% of trained voices, 5% of new ones), with 32 it starts to interpolate (47% of new ones), as the law predicts for any input path.

**More real speakers yield less and less.** By the fitted law, with K = 128:

| N | New voices copied (prediction) |
|---|---|
| 8 | 44% (measured: 51% / 37%) |
| 32 | 55% (measured: 61% / 57%) |
| 128 | 57% |
| infinity | 58% |

Going from 32 to 128 speakers (4 times more data) would yield about 2 points. If the real decay is slower than 1/N, the gain is larger, but it remains expensive. This is the riskiest prediction of the theory, because it extrapolates beyond N = 32.

**The floor `a` = 0.42 is the remaining problem.** Even with infinitely many speakers, 4 out of 10 samples of new voices would not copy the voice, against 1 in 10 for trained voices. The difference is the speaker-specific component, which for a new voice has to come entirely from a single short reference passing through a 32-dimensional global vector.

## 5. Solutions that follow from the theory

The law says what counts: (text, voice) pairs seen by the decoder, with N (distinct voices) weighing more than K. Three paths:

**(a) Fill the rectangle with pseudo-speakers (tested; did not work, section 6).** There is a cheap transformation that changes the voice and keeps the text: playing the recording faster or slower by a factor r raises or lowers the pitch and the formants by r. Applied to the long row (the 2,000 sentences of the default voice) and to the short rows, it creates new voices with the same sentences, without recording anything. In speaker recognition it is common practice to treat speed-altered copies as new speakers. In the law, this increases N (and K) for free. The predictable limit: the pseudo-voices vary along only one direction of the voice space (vocal tract size/pitch); they count as speakers only to the extent that real voices differ along that direction.

Implementation: `scripts/coverage/augment_voices.py` adds, for each training sentence, a speed-altered copy shifted between 1.5 and 4 semitones up or down; variant `-aug` in `scripts/coverage/colab_job.sh`. The number of training steps is the same.

Predictions registered before the result (new voices copied, step 2,500, full style):

| Training run | Normal (measured, 2 seeds) | If the pseudo-voices do not count | If they count as speakers (effective N ≥ 32) |
|---|---|---|---|
| 8 × 8 | 26% / 39% | ~31% | ~50% |
| 8 × 128 | 51% / 37% | ~44% | ~55% |

Since a seed varies by ±5 points (up to 14 between seeds), only a gain of more than ~10 points over the normal mean counts as an effect.

**(b) Take the text out of the equation: speech in the default voice + voice conversion.** If `f(c, s) = V(T(c), s)`, with `T` reading the text in the default voice and `V` changing the voice without looking at the text, the interaction term vanishes by construction. `T` already works (first term of the decomposition). `V` does not need transcriptions, so it can be trained on thousands of speakers of untranscribed speech: N is no longer limited by the data with text. This is the structural path; it costs training a second model.

**(c) Lower the floor `a`.** The floor comes from summarizing a new voice in a global vector taken from one reference. Using several references from the same actor (in dubbing there is the whole film) and a code that varies along the sentence attack this term, not the coverage terms.

For dubbing, (b) and (c) fit the scenario: there is a lot of audio of the original actor without aligned transcription, and no training text in their voice.

## 6. Result of the test of (a): the pseudo-voices do not count as speakers

Two training runs with pseudo-speakers (`n8_k8_s1-aug`, `n8_k128_s1-aug`; each training sentence gets a speed-altered copy, 41 and 54 pseudo-speakers in total), same number of steps and same evaluation. Step 2,500, full style:

| Training run | Trained voices copied | New voices copied | WER new voices | Prediction "do not count" | Prediction "count" |
|---|---|---|---|---|---|
| 8 × 8 normal (2 seeds) | 45% / 44% | 26% / 39% | 14% / 24% | | |
| 8 × 8 with pseudo-speakers | 50% | **28%** | 14% | ~31% | ~50% |
| 8 × 128 normal (2 seeds) | 91% / 88% | 51% / 37% | 8% / 9% | | |
| 8 × 128 with pseudo-speakers | 84% | **42%** | 8% | ~44% | ~55% |

Both results fall on the prediction "the pseudo-voices do not count" (28% against ~31%; 42% against ~44%) and within the range of the two normal seeds. At step 1,000 the picture is the same (22% against 24%; 36% against 30%). Solution (a) does not work. It is one seed per configuration, but the gap to the "count" prediction (22 and 13 points) is larger than the variation between seeds.

What this says about the theory:

- The two-level law predicted both numbers with N = 8, without refitting. The N that enters it is the number of **real** voices; fifteen to twenty times more speaker labels changed nothing.
- The limit anticipated in section 5 is what happened: changing the speed shifts the voice along only one direction (pitch and formants together). The new voices in the test differ from the trained ones along many directions (timbre, glottal source, articulation, rhythm), and the decoder needs to have seen variation along those directions.
- The term A/N therefore counts the real variety of voices, not (text, voice) pairs in general. Filling the rectangle with simple signal transformations does not replace speakers.

Practical consequence: paths (b) and (c) remain. Path (b) is the one that attacks the term A/N directly, because voice conversion trains without transcriptions and can use thousands of real speakers; path (c) attacks the floor `a`. A stronger version of (a) would be to generate the pseudo-voices with a voice converter already trained on many speakers (the 2,000 sentences of the default voice spoken in hundreds of real voices), which is path (b) used as data augmentation.

## 7. Next test: voice conversion (path b), with registered predictions

Two experiments with a pre-trained voice converter (kNN-VC: replaces the WavLM features of the audio with the closest ones among those of the target speaker; it does not use text and is not trained here).

**7.1 Cascade: read in the default voice, then convert** (`scripts/coverage/vc_cascade.py`). The English model, without any training on other speakers, reads the 25 sentences in the default voice; the converter takes each audio clip to the reference voice, described by R = 1, 4, 16 or 64 sentences from the same speaker. The references now include 4 "external" speakers (LibriTTS test-clean), whom neither our models nor the converter have heard.

Predictions:

- The WER stays close to that of the default voice (up to ~8%) in all groups, because the text × voice interaction term vanishes.
- Voice copying does not depend on whether the voice is trained, new or external.
- Copying rises with R (path c: more reference lowers the floor). With R ≥ 16, it exceeds 58% on the new voices, the ceiling the law gives the current model with infinitely many speakers. If with R = 64 it stays below 58%, path (b) with this converter fails.

**7.2 Real voices via conversion in training** (`scripts/coverage/vc_voices.py`, variant `-vcaug`). The 2,000 sentences of the default voice get a copy converted to 200 real LibriTTS speakers (10 sentences per voice; speakers outside all training runs and all references; only their audio is used). This is the test of what section 6 concluded: if the term A/N counts real variety of voices, these copies should count, unlike the speed-altered ones.

| Training run | Normal (2 seeds) | If the converted voices do not count | If they count (N ≈ 208) |
|---|---|---|---|
| 8 × 8 | 26% / 39% | ~31% | ~55% |
| 8 × 128 | 51% / 37% | ~44% | ~57% |

The criterion is the same: only more than ~10 points above the normal mean counts as an effect.

**Status of 7.1 and 7.2:** the scripts are in the repository but have not been run. Running the third-party voice converter was blocked by the permission controls of the environment in which the experiments are launched; it depends on the author's authorization.

## 8. Two tests with our model only, with registered predictions

**8.1 The risky prediction: 128 speakers.** Training runs `n128_k8_s1` and `n128_k32_s1` (the 32 speakers of the grid plus 96 from the same corpus; same new voices, same steps). The law fitted in section 3 predicts, for new voices copied, 56% (K = 8) and 57% (K = 32): almost nothing above the 32 speakers (measured: 51% / 47% with K = 8; 61% / 57% with K = 128). For trained voices, 52% (K = 8) and 82% (K = 32). If the real decay is slower than 1/N, the new voices exceed 67% (prediction + 10 points) and the law is wrong at this point, which would be good news: more speakers would still pay off.

**8.2 Several references per voice (path c).** In evaluation, the style code becomes the mean of the codes of R = 4 and 16 sentences from the same speaker (`--n-refs` in `scripts/eval/synth_check.py`), without training anything. The voice measure gains a second column, the similarity to the mean of 20 sentences of the speaker (`sim_spk`), comparable to that of the default voice. Two hypotheses for the floor `a`:

- the floor is noise from estimating the voice from a single sentence: copying of new voices rises by 10 points or more from R = 1 to R = 16;
- the floor belongs to the decoder, which cannot produce the voice even with the right code: nothing changes (or it gets worse, if the mean pulls the code toward the default voice).

The references now include 4 external speakers (LibriTTS test-clean), in addition to the usual 8 trained and 4 new ones.

**8.3 Result** (step 2,500, full style; one seed per training run; `sim_ref > sim_default` as in the previous sections):

| Training run | Trained copied (predicted → measured) | New copied (predicted → measured) | External copied | WER trained / new / external |
|---|---|---|---|---|
| 128 × 8 | 52% → **52%** | 56% → **55%** | 86% | 12% / 8% / 13% |
| 128 × 32 | 82% → **76%** | 57% → **74%** | 86% | 12% / 8% / 16% |

- With K = 8 the law got both numbers right to within 1 point.
- With K = 32 the new voices came out 17 points above the prediction, beyond the 67% limit I had set. At this point the law was wrong: with 128 speakers, more sentences per speaker still pay off, and the 58% ceiling does not hold. It is the best result of the project for new voices (74% copied with 8% WER; the previous best was 61% / 57% with 32 × 128).
- The external voices (another set of recordings) are copied more than the new ones from the same corpus. They are farther from the default voice (lower `sim_default`), which makes the criterion easier to meet; it is not a sign that they are easier to imitate.

Redoing the fit with the 14 training runs, the law a + A/N + B/(NK) gives a = 0.39 and RMSE 0.063 (previously 0.050), and still predicts 60% for 128 × 32. Square-root laws, such as a + A/√N + C/√K, fit the 12 old training runs equally well (RMSE 0.052) and, fitted on those alone, predicted 58% and 65% for the two new training runs; in them the floor `a` drops to 0.16. Honest conclusion: **the data identify neither the floor nor the shape of the decay**. What holds is the two-level structure (N for new voices, K for trained ones, and K also helping the new ones when N is large). The statement "more speakers yield less and less" from section 4 is on hold until the replicate (section 8.5).

**8.4 Result of the several references.** Voices copied by the per-speaker measure (`sim_spk > sim_default`), which is the appropriate one when the code comes from several sentences:

| Training run | Group | R = 1 | R = 4 | R = 16 |
|---|---|---|---|---|
| 128 × 8 | trained | 66% | 66% | 66% |
| 128 × 8 | new | 47% | 67% | 67% |
| 128 × 8 | external | 77% | 83% | 90% |
| 128 × 32 | trained | 84% | 71% | 75% |
| 128 × 32 | new | 74% | 78% | 72% |
| 128 × 32 | external | 95% | 94% | 99% |

Neither of the two hypotheses is fully confirmed. With K = 8 the new voices rise by 20 points already with 4 references; with K = 32, where copying was already high, nothing changes for the new ones and the trained ones drop. The mean similarity to the speaker rises in all groups (for example 0.713 → 0.743 for the new ones with K = 32), but the similarity to the default voice rises with it, because the mean of the codes lies closer to the center. The WER of the external voices drops (16% → 9% with K = 32; 13% → 10% with K = 8) and that of the others stays the same. By the per-sentence measure (`sim_ref`), copying drops with R, as expected: the code is no longer that of that sentence. Reading: averaging codes helps when the model copies poorly and makes speech more stable, but it is not a general gain; a consistent gain would require training with the speaker's mean code, not just using it at test time.

**8.5 Replicate and 200 speakers: predictions.** `n128_k32_s2` (same configuration, another seed) and `n200_k32_s1` (200 speakers). If the 74% is real, the replicate lands at 64% or more; if it is seed luck, near 60%. For 200 × 32, the refitted 1/N law predicts 60% and the square-root law predicts 68%.

**8.6 Result of the replicate and the 200 speakers.** Step 2,500, style at 100%:

| Training run | Trained copied | New copied | External copied | WER trained / new / external |
|---|---|---|---|---|
| 128 × 32, seed 1 | 76% | 74% | 86% | 12% / 8% / 16% |
| 128 × 32, seed 2 | 66% | 56% | 86% | 13% / 8% / 11% |
| 200 × 32, seed 1 | 66% | 64% | 91% | 12% / 7% / 11% |

- **The 74% was largely seed luck.** The replicate gave 56%, below the registered threshold of 64%; the mean of the two seeds is 65%. The 18-point difference between seeds is the largest in the project (previously, 14).
- **200 speakers do not beat 128.** 64%, between the two predictions (60% and 68%) and equal to the mean of 128 × 32. Neither of the two laws is rejected.
- Refitted on the 16 training runs, the two laws become equivalent: a + A/N + B/(NK) with a = 0.39 (RMSE 0.060) and a + A/√N + C/√K with a = 0.19 (RMSE 0.056). For 128 × 32 and 200 × 32 they predict 60% and 65–66%; they only truly diverge for N in the thousands (61% against 70% with 1,000 speakers).
- **What limits the conclusion now is the measurement, not the training.** With 4 new voices × 25 sentences, the noise between seeds (up to 18 points) is larger than the difference between the laws (5 points). To separate the laws, each training run must be evaluated on dozens of new voices (LibriTTS has plenty of speakers outside training), which costs only evaluation.
- The statement "more speakers yield less and less" from section 4 holds again as a description: from 32 to 128 speakers there is a gain with K = 32 (52% → 65%), from 128 to 200 there is no measurable gain.

Several references (R = 4), by the per-speaker measure, in the three new training runs and the two from section 8.4:

| Training run | Trained R = 1 → 4 | New R = 1 → 4 | External R = 1 → 4 |
|---|---|---|---|
| 128 × 8 | 66% → 66% | 47% → 67% | 77% → 83% |
| 128 × 32, seed 1 | 84% → 71% | 74% → 78% | 95% → 94% |
| 128 × 32, seed 2 | 73% → 68% | 59% → 69% | 74% → 97% |
| 200 × 32 | 77% → 68% | 57% → 75% | 76% → 90% |

With the new training runs the picture becomes clearer than in section 8.4: the mean of 4 references **raises the copying of new voices in all four training runs** (+4 to +20 points, reaching 67–78%) and of external voices in three out of four, and **lowers that of trained voices** in three out of four. This is the prediction of the decomposition in section 2: the mean removes the part of the code that belongs to the sentence and not to the speaker, which helps where the model has not memorized the voice and hurts where it has. As a practical solution, it is the only one of those tested that improves new voices without training anything.

**8.7 Wide evaluation: predictions.** Three training runs with K = 32 (N = 8, 32 and 128; suffix `-wide`) evaluated on all LibriTTS test-clean voices with enough sentences (up to 39 speakers, 25 sentences each, about 10 times more new voices than before), with R = 1 and R = 4. External voices are copied more easily than the 4 "new" ones of the grid (86–91% against 56–74%), so the prediction is about the shape, not the level. Registered before running:

- By the 1/N law, the non-copied fraction drops by 13 points from N = 8 to 32 and by 3 points from 32 to 128 (ratio 0.25). By the square-root law, it drops by 18 and 9 points (ratio 0.5). If the gain from 32 to 128 is below one third of the gain from 8 to 32, the 1/N law stands; if it exceeds 40%, the square-root law does.
- With R = 4, per-speaker copying rises on the external voices in all three training runs.

**8.8 Result of the wide evaluation.** Step 2,500, style at 100%, 38 LibriTTS test-clean speakers × 25 sentences (950 samples per training run), none of them heard by any model:

| Training run | External copied (95% CI) | WER external | Per speaker, R = 1 → 4 | 4 new voices of the grid |
|---|---|---|---|---|
| 8 × 32 | 57% (43–70) | 13% | 66% → 64% | 36% |
| 32 × 32 | 70% (58–81) | 12% | 81% → 79% | 48% |
| 128 × 32 | 79% (68–89) | 12% | 88% → 84% | 75% |

The intervals come from resampling the 38 speakers (10,000 times); the differences between training runs are paired on the same speakers.

- **The gain from 32 to 128 speakers is real**: 9 points (95% CI from 1.5 to 17), against 13 points from 8 to 32 (4 to 23). The ratio between the two gains is 0.7; in 80% of the resamples it exceeds 0.4 (criterion of the square-root law) and in 14% it falls below 1/3 (criterion of the 1/N law). ~~The square-root law comes out favored~~ (revised in section 8.10: the laws cannot be told apart). This reverses the reading of sections 4 and 8.6 that more speakers yield less and less; that reading came from the noise of evaluating on 4 voices.
- The WER of the external voices does not change with N (12–13%): more speakers improve copying at no cost in intelligibility.
- **The prediction about several references failed.** On the 38 external voices the mean of 4 references does not raise per-speaker copying in any of the three training runs (it drops 2 to 4 points). The gain seen in sections 8.4 and 8.6 came from the 4 new voices of the grid (here again 34% → 62% with N = 32) and does not generalize; what averaging consistently does is lower the WER (13% → 12%, 12% → 10%, 12% → 9%). The "training-free solution" of section 8.6 is therefore downgraded to a speech stabilizer.
- The 4 new voices of the grid are harder than average (36–75% against 57–79%) and, on their own, give the right order but unstable values. The 128 × 32 training run with seed 1 repeated on them the value from the first time (75% against 74%): the difference from seed 2 (56%) comes from the seed, not from evaluation chance.
- Per external speaker, the number of voices copied in more than half of the sentences goes from 21 to 27 and 30 (out of 38). In every training run there are speakers at 0% and at 100%: copying is almost all-or-nothing per voice.

One execution failure: the first 8-speaker training run was evaluated without the external voices (the download failed) and re-evaluated later from the same checkpoint; the numbers above are from the re-evaluation.

**8.9 Which voices fail, and why** (`scripts/coverage/voice_hardness.py`, no new training; rank correlations over the 38 external speakers):

- **The difficulty belongs to the voice, not to chance:** the per-speaker copy rate correlates 0.75 between N = 8 and 32, 0.74 between 32 and 128, and 0.59 between 8 and 128.
- **High-pitched voices fail more:** correlation of −0.53, −0.52 and −0.37 between copying and the F0 of the reference. The default voice (LJSpeech) is female; voices close to it are harder to "beat" under the measure `sim_ref > sim_default`. Copying rises with the distance (in the latent) from the reference to the default voice: 0.47, 0.58 and 0.53. Part of the "not copied" is therefore the voice being pulled toward the default voice, and part may be an artifact of the measure; separating the two requires the continuous margin or a listening test.
- **The distance to the nearest trained voice predicts failure with few speakers and stops predicting it with many:** −0.33 (N = 8), −0.40 (N = 32), +0.13 (N = 128). This is what the "coverage" reading predicts: with few speakers, the decoder only reproduces voices close to those it has heard; with 128, almost every voice has a neighbor, and the failure that remains is the attraction toward the default voice.

Joint reading with sections 8.4–8.8: the bottleneck is not the encoder describing the new voice (the ECAPA vector, trained on thousands of voices, does not help; averaging references does not help), but the decoder being able to **render** it. This is a hypothesis about the architecture, and it is testable.

**8.10 Critical review, no new training** (full reports in `reports/theory/02_derivation.md`, `03_information.md` and `04_data.md`; scripts in `reports/theory/analyses/`). Corrections to what is above:

- **The exponent 0.26 has no statistical support.** It is what two parameters do when passed through three single-seed points. Simulating the design with the measured variance, a true exponent of 0.26 is estimated anywhere between 0.06 and 0.53, and the right law is chosen in 16–62% of cases. The "80% in favor of the square-root law" (8.8) and the extrapolations to 247 and 1,000 speakers are withdrawn. On the 4-voice grid, the square-root law wins only because of the training runs with N = 2.
- **The measure distorts the difficulty of the voices.** With normalized vectors, the margin is exactly m = (½ − ρ)·‖e_ref − e_0‖², with ρ the attraction of the output toward the default voice; voices similar to the default voice have a tiny margin and switch sides due to noise. Pitch and similarity to the default voice explain 28–59% of the voice difficulty in the binary measure and 2–9% in the direct similarity to the reference. Without the default voice in the measure, the apparent exponent rises to 0.41–0.55 (same caveat of few points).
- **Voice bits do not grow as log N** (section 8.8 and the proposal's table): the Gaussian estimate stays near 17 bits on the latent mean, stable from N = 32 to 200; the bias is small, but it measures a continuous voice vector. The intrinsic dimension of the per-speaker means is 7–10, which rules out "failure ∝ distance" (would require d ≈ 4) and is compatible with "failure ∝ distance²" (d ≈ 7.7), without being a test.
- **What holds:** trained voices follow only K (collapse K^0.46, N effect null ± 0.07); copying is almost all-or-nothing per voice (ICC 0.83–0.94); the variation comes from the reference voice, not from the sentence; the mean of 4 references does not help on the 38 voices (−2.5 points, CI [−6.4, +0.8]); the gain from 32 to 128 speakers is real.
- **Coverage model + attraction toward the default voice:** copied ⇔ K·Σ_i k(v, v_i) > M·k(v, v_0), with M = 2,000 sentences of the default voice and voice spectrum λ_j ∝ 1/j (the latent measures 1.05–1.1). It reproduces the curve, the signs of the correlations and the grid (r = 0.97), but with 5 parameters chosen on 3 points; and it predicts that the pseudo-speakers (section 6) should have counted, which did not happen.

Predictions registered now, before any training:

1. Half of the default-voice sentences (M = 1,000) has the same effect on new voices as doubling K.
2. An 8-dimensional latent does not change copying; with 2, it drops. The rate can go down to 25–30 bits per sentence without loss.
3. N = 2,456, K = 32: about 9% of voices not copied under coverage, against 16% or more under the laws with a floor (requires at least 200 evaluation voices).
4. Two stages: copying follows N_voice as long as the units leak little voice; a leak of weight 0.2 imposes a floor of about 15%.

Recommended design for the next training runs (power simulation): 60–100 fixed voices × 10 sentences, N ∈ {8, 32, 128, 512, 2,048}, 3–5 seeds per N, a continuous metric without the default voice as the main one, synthesis with zeroed style per voice, and double resynthesis of each checkpoint.

**8.11 Information theory: where the voice bits are lost.** Two theories proposed by independent agents (coverage as source coding; copying as a Bayesian hypothesis test), each critiqued in three rounds (reports v1–v3 and scripts in `reports/theory/information/A` and `B`). What survived, measured without a model on the generated audio of the 38 external voices (C = I(output voice; reference), bounds from the identification confusion matrix; ceiling log2 38 = 5.25 bits):

| Link in the chain | N = 8 | N = 128 |
|---|---|---|
| Real speech (verifier) | 4.2–4.7 | 4.2–4.7 |
| μ (Gaussian model) | 5.1 | 5.2 |
| z (Gaussian model) | 4.4 | 4.8 |
| Output, Fano bound | 0.3 | 0.9–1.1 |
| Output, cross-entropy bound | 0.75 [0.28, 1.20] | 1.99 [1.35, 2.60] |

- **The bottleneck is the decoder:** it loses about 4 bits; the encoder and the latent are not the problem.
- **The gain with N is coverage, not precision:** H(Ŷ) rises 1.44 bit [0.68, 2.32] from N = 8 to 128, H(Ŷ|S) does not change (+0.07 [−1.24, 1.54]). The coverage theory reaches the same conclusion by another route: a decoder "temperature" of about 20, common to the three training runs (19–27; a common value is not rejected).
- **Law in bits:** 0.31 bit per doubling of N [0.17, 0.47], with two points. Predictions: C(512) = 2.6 [1.9, 3.5], C(2,048) = 3.2 [2.3, 4.5]; matching real speech would require about 10^5 speakers (10^3.6–10^8.2). Falsified if: C(2,048) < 2.25.
- **Negatives in one number:** ECAPA alone transmits 0.00 bit; averaging references changes C between −0.40 and +0.31 with no direction; style at 60% lowers C in 8 of 10 training runs; rank correlation between non-copied and C across 12 training runs: −0.49.
- **A priori prediction confirmed:** longer reference sentences make copying more all-or-nothing (+0.035 [0.015, 0.053], p = 0.0002).
- **Died:** parameter-free optimal Bayes (uses 22× more evidence than the real decoder); trained voice as a noisy point; "information per voice" as a predictor; stable exponent; per-voice capacity growing with N. The attraction toward the default voice survives with an effective mass of 2,300–5,000 sentences (the 2,000 real ones); the effect of cutting them in half (4–9 points) requires about 6 training runs per condition.
- **Consequence:** the architectural target is to increase H(Ŷ), the set of voices the decoder can reach, with more voices in the voice stage (untranscribed audio) or with conditioning in all decoder layers (hypothesis, experiment E5 of the proposal). Each new training run should measure C, H(Ŷ) and H(Ŷ|S) on 60–100 voices and keep the generated audio.

**8.12 Experiments ready in the code, with predictions registered before running.** Without a GPU, they were implemented and tested on CPU (small model: training step, inference and checkpoint loading):

- `-film` (`--style-film`): the style vector also modulates, via FiLM, the states of the decoder's two LSTMs at each step. Initialized at zero, so the pre-trained model starts out identical (checked). **Prediction:** if the conditioning mechanism is the bottleneck, C (cross-entropy bound, 38 external voices) rises by at least 0.5 bit at N = 8 and at N = 128, with H(Y|S) equal or lower; if C stays within the interval of the normal training run, the bottleneck is not the style's point of entry.
- `-z8` / `-z2` (`--z-latent-dim`): latent of 8 or 2 dimensions, with the VAE heads re-initialized when loading the 32-dimensional checkpoint. **Prediction** (section 8.10): copying and C unchanged with 8; they drop with 2.
- `-lj1000`: half of the default-voice sentences, with the same test sentences. **Prediction** (section 8.11): 4 to 9 points fewer non-copied voices at N = 128; separating the theories requires about 6 training runs per condition.
- Every evaluation now saves the speaker vectors of all samples (`embeddings.npz`) and computes the voice bits per reference group (`scripts/coverage/voice_bits.py`: C by the cross-entropy and Fano bounds, H(Y), H(Y|S), intervals by resampling the voices). Checked on synthetic data: perfect copying gives 5.25 out of 5.25 bits; no copying gives 0.00.

## 9. Limits of this treatment

- The law was fitted after seeing the data; its honest test is the predictions of sections 4 and 5.
- 10 configurations, N ≤ 200, 4 new voices per evaluation (noise between seeds of up to 18 points); the wide evaluation of section 8.8 has 38 voices, but one seed and only K = 32, and one voice measure (one speaker encoder).
- The measure is binary per sample (voice closer to the reference than to the default voice); it does not measure naturalness.
- The decomposition of section 2 is about the expected error; it was not measured term by term.
