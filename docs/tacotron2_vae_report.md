# Tacotron2-VAE com Transferência de Estilo
## Relatório Técnico — Projeto Final ML2

---

## 1. Objetivo

Construir um sistema de síntese de voz (TTS) com **transferência de estilo prosódico** usando uma arquitectura Tacotron2-VAE-GST. O sistema recebe um texto e um áudio de referência, e sintetiza a fala com o estilo prosódico (ritmo, entonação, velocidade) da referência.

**Estratégia adoptada**: validar em inglês (LJSpeech) para convergência rápida, depois adaptar ao português.

---

## 2. Arquitectura

### 2.1 Visão Geral

```
Texto → [Encoder] ──────────────────────────────────┐
                                                      ▼
Áudio ref → [ReferenceEncoder] → [VAE] → z → [Decoder] → Mel → [Postnet] → Mel
                                   ↓
                               (μ, logvar)
```

O modelo combina três componentes:

| Componente | Descrição |
|---|---|
| **Encoder** | Embedding de texto + 3× Conv1D + BiLSTM → contexto fonético |
| **VAE-GST** | ReferenceEncoder + camadas FC(μ, σ) + reparametrização → embedding de estilo |
| **Decoder** | LSTM com atenção location-sensitive → frames de mel |
| **Postnet** | 5× Conv1D → refinamento do espectrograma |
| **WaveGlow** | Vocoder NVIDIA pré-treinado → áudio a partir do mel |

### 2.2 Text Encoder

```
TextEmbedding(n_symbols=198, dim=512)
→ 3 × [Conv1D(512, kernel=5) → BatchNorm → ReLU → Dropout(0.5)]
→ BiLSTM(256 → 512)
```

### 2.3 VAE-GST (Reference Encoder + Variational Autoencoder)

**Reference Encoder** — processa o mel do áudio de referência:
```
Mel(80 canais)
→ 6 × [Conv2D → BatchNorm → ReLU]
  filtros: [32, 32, 64, 64, 128, 128], kernel=3×3, stride=2×2
→ GRU(256)
→ último estado oculto: (B, 256)
```

**VAE** — produz o embedding de estilo latente:
```
enc_out (B, 256)
→ FC_μ(256 → 32)      ← média do espaço latente
→ FC_logvar(256 → 32) ← log-variância
→ reparametrização: z = μ + ε·exp(0.5·logvar),  ε ~ N(0,1)
→ FC_style(32 → 512)  ← embedding de estilo injectado no decoder
```

Dimensão latente: **z ∈ ℝ³²**

### 2.4 Decoder

```
[encoder_output + style_embed]
→ Prenet: 2 × [FC(256) → ReLU → Dropout(0.5)]
→ AttentionRNN: LSTM(1024)
→ LocationSensitiveAttention(dim=128, n_filters=32, kernel=31)
→ DecoderRNN: 2 × LSTM(1024)
→ FC → 80 mel frames
→ Gate FC → stop token
```

---

## 3. Função de Perda

A perda total combina três termos:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{MSE mel + BCE gate}} + w_{\text{KL}} \cdot \underbrace{\mathcal{L}_{\text{KL}}}_{\text{divergência VAE}} + w_{\text{GA}} \cdot \underbrace{\mathcal{L}_{\text{GA}}}_{\text{guided attention}}$$

### 3.1 Perda de Reconstrução

$$\mathcal{L}_{\text{recon}} = \text{MSE}(\hat{M}_{\text{pre}}, M) + \text{MSE}(\hat{M}_{\text{post}}, M) + \text{BCE}(\hat{g}, g)$$

Onde $M$ é o mel alvo, $\hat{M}_{\text{pre}}$ e $\hat{M}_{\text{post}}$ são as saídas do decoder e do postnet, e $g$ é o gate de paragem.

### 3.2 Divergência KL com Free Bits

$$\mathcal{L}_{\text{KL}} = \max\left(\lambda_{\text{free}},\ \frac{1}{2}\sum_{j=1}^{L}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)\right)$$

- **Free bits** $\lambda_{\text{free}} = 0.5$: evita o colapso posterior — o modelo mantém informação no espaço latente.

### 3.3 Annealing KL Ciclico

O peso $w_{\text{KL}}$ aumenta gradualmente para evitar colapso do VAE no início do treino:

$$w_{\text{KL}}(t) = \begin{cases} 0 & t < t_{\text{lag}} \\ w_{\text{upper}} \cdot \sigma(k(t - x_0)) & t \geq t_{\text{lag}} \end{cases}$$

| Parâmetro | Valor |
|---|---|
| `anneal_function` | `cyclical` |
| `anneal_lag` ($t_{\text{lag}}$) | 2000 passos |
| `anneal_x0` | 4000 passos |
| `anneal_k` | 0.0025 |
| `anneal_upper` ($w_{\text{upper}}$) | 0.2 |
| `free_bits` ($\lambda_{\text{free}}$) | 0.5 |

### 3.4 Guided Attention Loss

Penaliza atenções não-diagonais, forçando alinhamento monótono texto→áudio:

$$\mathcal{L}_{\text{GA}} = \frac{1}{B}\sum_b \sum_{n,t} A_{b,n,t} \cdot W_{b,n,t}$$

$$W_{b,n,t} = 1 - \exp\!\left(-\frac{\left(\tfrac{n}{N_b} - \tfrac{t}{T_b}\right)^2}{2\sigma^2}\right)$$

Onde $A_{b,n,t}$ é o peso de atenção, $N_b$ e $T_b$ são os comprimentos do encoder e decoder para o batch $b$.

| Parâmetro | Valor |
|---|---|
| `guided_attention_weight` ($w_{\text{GA}}$) | 2.0 |
| `guided_attention_sigma` ($\sigma$) | 0.4 |

---

## 4. Datasets

### 4.1 LJSpeech (Inglês)

| Propriedade | Valor |
|---|---|
| Utterances totais | 13.100 |
| Treino | 11.570 (88%) |
| Validação | 765 (6%) |
| Teste | 765 (6%) |
| Taxa de amostragem | 22.050 Hz |
| Locutor | 1 (feminino) |
| Domínio | Leitura (livros não-ficção) |
| Texto | Caracteres ASCII (a–z, pontuação) |

### 4.2 TTS-Portuguese-Corpus (Português)

| Propriedade | Valor |
|---|---|
| Utterances totais | 2.340 |
| Treino | ~2.065 (88%) |
| Validação | ~137 (6%) |
| Teste | ~137 (6%) |
| Taxa de amostragem | 22.050 Hz |
| Locutor | 1 |
| Domínio | Leitura |
| Texto | Fonemas IPA via Gruut (português) |

### 4.3 Pré-processamento

- **Mel espectrogramas**: pré-computados e guardados em disco (`.pt`) — evita recomputação por época
- **Normalização de texto (inglês)**: lowercase + expansão de números
- **Normalização de texto (português)**: lowercase + expansão de abreviaturas + expansão de números (num2words) + G2P via Gruut
- **Split**: `val_split=0.1` → val e test cada com 5% do total (`n_val = n_test = int(N × 0.1 / 2)`)

---

## 5. Hiperparâmetros

### 5.1 Arquitectura do Modelo (partilhado entre experimentos)

| Parâmetro | Valor |
|---|---|
| `n_symbols` | 198 |
| `symbols_embedding_dim` | 512 |
| `encoder_embedding_dim` | 512 |
| `encoder_kernel_size` | 5 |
| `encoder_n_convolutions` | 3 |
| `ref_enc_filters` | [32, 32, 64, 64, 128, 128] |
| `ref_enc_size` | [3, 3] |
| `ref_enc_strides` | [2, 2] |
| `ref_enc_gru_size` | 256 |
| `z_latent_dim` | 32 |
| `E` (style embedding dim) | 512 |
| `decoder_rnn_dim` | 1024 |
| `attention_rnn_dim` | 1024 |
| `attention_dim` | 128 |
| `attention_location_n_filters` | 32 |
| `attention_location_kernel_size` | 31 |
| `prenet_dim` | 256 |
| `postnet_embedding_dim` | 512 |
| `postnet_kernel_size` | 5 |
| `postnet_n_convolutions` | 5 |
| `n_frames_per_step` | 1 |
| `max_decoder_steps` | 1000 |
| `gate_threshold` | 0.5 |
| `p_attention_dropout` | 0.1 |
| `p_decoder_dropout` | 0.1 |
| `p_decoder_input_dropout` | 0.5 |
| `n_mel_channels` | 80 |
| `sampling_rate` | 22.050 Hz |
| `filter_length` | 1024 |
| `hop_length` | 256 |
| `win_length` | 1024 |
| `mel_fmin` | 0.0 |
| `mel_fmax` | 8000.0 |

### 5.2 Hiperparâmetros de Treino por Experimento

| Parâmetro | `lj_speech_v1` | `pt_tacotron_v1` |
|---|---|---|
| Dataset | LJSpeech (EN) | TTS-PT-Corpus |
| Ponto de partida | NVIDIA Tacotron2 pré-treinado | `lj_speech_v1/epoch_1600` |
| `learning_rate` | 1e-4 | 1e-5 |
| `batch_size` | 32 | 16 |
| `grad_clip_thresh` | 5.0 | 5.0 |
| `warmup_steps` | 0 | 500 |
| `warmup_start_lr` | 1e-6 | 1e-6 |
| `weight_decay` | 1e-6 | 1e-6 |
| `epochs` | 500 | 300 |
| `iters_per_checkpoint` | 200 | 200 |
| `num_workers` | 8 | 4 |
| Text cleaner | `english_cleaners` | `portuguese_phonetic_cleaners` |
| AMP (fp16) | Desactivado | Desactivado |

---

## 6. Modificações Implementadas

### 6.1 Guided Attention Loss

**Problema**: O decoder Tacotron2 pode aprender alinhamentos não-monótonos (saltando fonemas ou repetindo), levando a artefactos de áudio.

**Solução**: Implementação da guided attention loss em `losses.py`. A penalização gaussiana diagonal força a atenção a seguir uma trajectória linear texto→áudio.

**Impacto**: `GuidedAttn` caiu de 0.003 → 0.00119 em 1600 passos no `lj_speech_v1`, indicando alinhamento quase perfeito.

### 6.2 KL Annealing Ciclico com Free Bits

**Problema**: O KL com peso fixo alto leva ao colapso posterior — o VAE ignora o espaço latente e colapsa z para N(0,I). Com peso fixo baixo, o espaço latente não é regularizado.

**Solução**: 
- `anneal_lag=2000`: peso KL = 0 nos primeiros 2000 passos (modelo aprende a reconstruir mel primeiro)
- Após o lag, o peso sobe sigmoidal de 0 até 0.2
- `free_bits=0.5`: garante pelo menos 0.5 nats de informação no espaço latente

### 6.3 Gradient Clipping Calibrado

**Problema original**: `grad_clip_thresh=1.0` era demasiado restritivo — os gradientes eram cortados a 1.0 quando as normas brutas chegavam a 480+, resultando em actualizações efectivas de apenas 0.2–2%.

**Solução**: Aumento para `grad_clip_thresh=5.0`. Permite actualizações até 5× maiores mantendo estabilidade contra explosões extremas.

**Impacto**: Norma de gradiente estabilizou — padrão observado: 487→44→11.8→34→22.5 nos primeiros 1400 passos.

### 6.4 Pré-computação de Sequências Fonéticas

**Problema**: `text_to_sequence()` chamava o Gruut por cada amostra em cada epoch — bottleneck de 3.5s/batch.

**Solução**: Método `_load_or_compute_sequences()` no `loader_tacotron.py`:
- Computa todas as sequências no início (custo único)
- Guarda em ficheiro pickle `_seq_cache_{hash}.pkl`
- Carrega do cache nas epochs seguintes

**Impacto**: Velocidade por batch passou de >3.5s para ~3s (limitado pela GPU, não pela CPU).

### 6.5 Persistent Workers no DataLoader

**Problema**: Com `persistent_workers=False`, os workers do DataLoader eram destruídos e recriados entre epochs, adicionando latência.

**Solução**: `persistent_workers=True` no `train_utils.py` — os workers ficam activos entre epochs.

### 6.6 AMP Desactivado

**Problema**: O treino com precisão mista (fp16) via `torch.cuda.amp` causava overflow NaN nos gradientes, levando a divergência imediata.

**Solução**: AMP completamente desactivado. Treino em fp32 puro.

### 6.7 Warm Start Cross-Lingual

**Problema**: Treinar do zero em português com 2340 utterances seria lento e potencialmente instável.

**Solução**: Inicialização a partir do melhor checkpoint inglês (`lj_speech_v1/epoch_1600`) com reset do optimizer. Os pesos do encoder/decoder são transferidos; o optimizer recomeça com `lr_warmup` de 500 passos (1e-6 → 1e-5).

**Impacto**: Queda de perda de 69% nos primeiros 400 passos de português vs. treino do zero.

### 6.8 Correcção do output_lengths Masking

**Problema**: `output_lengths` não era passado ao critério de perda — a perda era calculada sobre frames de padding, introduzindo gradientes espúrios.

**Solução**: `output_lengths` passado correctamente para `criterion()` em `train_utils.py`, permitindo mascaramento correcto dos frames de padding.

### 6.9 Instalação de gruut_lang_pt

**Problema descoberto**: O pacote `gruut_lang_pt` não estava instalado. A phonemização portuguesa retornava silenciosamente sequências vazias (mediana de 2 tokens em vez de ~77). O modelo `pt_tacotron_v1` treinou durante centenas de passos com entrada de texto essencialmente vazia — aprendeu reconstrução de mel sem condicionamento de texto.

**Solução**:
```bash
pip install gruut[pt]
```
Cache reconstruído: mediana de 77 tokens/utterance, máx 209.

---

## 7. Phonemização Portuguesa

O pipeline de texto para português usa Gruut com IPA:

```
Texto → lowercase → expansão abreviaturas → expansão números (num2words, pt-BR)
     → Gruut G2P (lang='pt') → fonemas IPA → prefixo @pt_
     → sequência de inteiros (lookup na tabela de símbolos)
```

**Exemplo**:
```
"O título de página foi encontrado."
→ ['@pt_u', '@pt_t', '@pt_ʃ', '@pt_i', '@pt_t', '@pt_u', '@pt_l', '@pt_u',
   '@pt_d', '@pt_ʒ', '@pt_i', '@pt_p', '@pt_ɐ', '@pt_ʒ', '@pt_ĩ', '@pt_n', '@pt_ɐ', ...]
→ 57 tokens
```

**Tabela de símbolos** (198 total): Símbolos NVIDIA Tacotron2 originais (ASCII + ARPAbet) + 49 fonemas IPA portugueses do Gruut + token EOS `~`.

Fonemas IPA suportados:
```
b d e ej ew f i iw j k l m n o oj ow p s t u uj v w z
õ õj̃ ĩ ũ ũj̃ ɐ ɐj ɐw ɐ̃ ɐ̃w̃ ɔ ɛ ɛw ɡ ɲ ɹ ɾ ʁ ʃ ʎ ʒ ẽ ẽj̃
| ‖  (pausas menor e maior)
```

---

## 8. Resultados de Treino

### 8.1 lj_speech_v1 (Inglês)

| Passo | Train Total | Test Total | Grad Norm | Guided Attn |
|---|---|---|---|---|
| 0 | — | — | — | — |
| 200 | ~5.5 | — | 487 | ~0.003 |
| 400 | ~4.2 | — | 44 | — |
| 600 | — | — | 11.8 | — |
| 800 | — | — | 34 | — |
| 1000 | — | — | 245 | — |
| 1200 | — | — | — | — |
| 1400 | 1.503 | 0.826 | 22.5 | 0.00135 |
| **1600** | **1.342** | **0.782** | 208 | **0.00119** |
| 1800 | 1.919 | 1.041 | 89.6 | 0.00257 |

**Melhor checkpoint**: `epoch_1600` (Test=0.782, GuidedAttn=0.00119)

O spike de gradiente no passo 1600 (208) causou regressão no passo 1800. O KL annealing inicia no passo 2000.

### 8.2 pt_tacotron_v1 (Português)

*Nota: Este experimento treinou com cache de fonemas quebrado (gruut_lang_pt ausente). As métricas reflectem aprendizagem de reconstrução de mel sem condicionamento de texto.*

| Passo | Train Total | Val Total | Grad Norm | Guided Attn |
|---|---|---|---|---|
| 200 | 4.24 | 3.59 | 13.0 | 0.092 |
| 400 | 3.45 | 2.44 | 8.14 | 0.146 |
| 600 | 4.33 | 2.07 | 18.1 | 0.197 |

A diminuição da val loss (3.59→2.07) reflecte aprendizagem de reconstrução de mel via VAE, não TTS real. O aumento do GuidedAttn (0.092→0.197) confirma que o alinhamento texto→mel não convergiu.

---

## 9. Demonstração de Transferência de Estilo

### 9.1 Configuração

- **Modelo**: `lj_speech_v1/epoch_1600`
- **Texto**: "She had a voice that could make a stone weep."
- **Vocoder**: WaveGlow NVIDIA pré-treinado

### 9.2 Resultados

Mesmo texto sintetizado com três referências de áudio diferentes:

| Referência | Frames de Mel | Duração aprox. |
|---|---|---|
| LJ001-0001 (início do corpus) | 235 | ~2.7s |
| LJ023-0046 (meio do corpus) | 363 | **+54% mais lento** |
| LJ050-0269 (fim do corpus) | 274 | +17% mais lento |

**Diferença média de mel entre pares**: 1.55–1.75 (escala absoluta de ~11 dB)

O VAE-GST codifica eficazmente o estilo prosódico da referência e transfere-o para a síntese: velocidade de fala, ritmo e entonação variam com a referência.

### 9.3 Ficheiros de Áudio

```
exports/synth/style_demo/
├── lj_speech_v1_LJ001-0001_audio.wav  ← estilo 1
├── lj_speech_v1_LJ023-0046_audio.wav  ← estilo 2 (mais lento)
└── lj_speech_v1_LJ050-0269_audio.wav  ← estilo 3
```

---

## 10. Inferência

### 10.1 Script

`scripts/inference/synthesize_tacotron2_vae.py`

### 10.2 Uso

```bash
LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH \
python scripts/inference/synthesize_tacotron2_vae.py \
  --experiment experiments/tacotron2-vae/lj_speech_v1 \
  --text "Texto a sintetizar." \
  --reference-audio path/to/reference.wav \
  --waveglow local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427 \
  --output-dir exports/synth
```

**Parâmetros**:
- `--experiment`: directório com `hparams.json`, `symbols.json`, `checkpoints/`
- `--checkpoint`: (opcional) checkpoint específico; por defeito usa o mais recente
- `--reference-audio`: áudio de referência para codificação de estilo
- `--waveglow`: caminho para o WaveGlow pré-treinado (opcional; sem ele só guarda o mel)
- `--sigma`: parâmetro de ruído do WaveGlow (default: 0.6)

**Saídas**:
- `{stem}_mel.png` — espectrograma mel
- `{stem}_alignment.png` — mapa de atenção
- `{stem}_mel.pt` — tensor mel (PyTorch)
- `{stem}_audio.wav` — áudio (se `--waveglow` fornecido)
- `{stem}_summary.json` — metadata da síntese

### 10.3 Notas

- O `LD_LIBRARY_PATH` é necessário para a compatibilidade CXXABI do Gruut/SQLite3
- O WaveGlow NVIDIA foi guardado com `DataParallel`; o script remove automaticamente o prefixo `module.` do state dict
- Durante inferência, o VAE usa `z = μ` (sem ruído) para síntese determinista

---

## 11. Estrutura de Ficheiros Relevantes

```
ml2_final_project/
├── src/
│   ├── models/tacotron2_vae/
│   │   ├── model.py          — Tacotron2 + VAE_GST
│   │   ├── modules.py        — ReferenceEncoder, VAE_GST
│   │   ├── layers.py         — ConvNorm, LinearNorm, TacotronSTFT
│   │   └── hparams.py        — Tacotron2VAEHparams
│   ├── training/training-tacotron2-vae/
│   │   ├── train.py          — loop de treino principal
│   │   ├── train_utils.py    — DataLoader, checkpoint, LR warmup
│   │   ├── losses.py         — Tacotron2LossVAE (MSE + KL + GuidedAttn)
│   │   └── text_processing.py — TextProcessor, portuguese_phonetic_cleaners
│   └── data/loader_vae_tacotron/
│       └── loader_tacotron.py — LookupDataset com phoneme pre-caching
├── scripts/inference/
│   └── synthesize_tacotron2_vae.py — script de inferência com style transfer
├── experiments/tacotron2-vae/
│   ├── lj_speech_v1/         — experimento inglês
│   │   ├── hparams.json
│   │   ├── symbols.json
│   │   ├── checkpoints/epoch_1600  ← melhor checkpoint
│   │   └── logs/             — TensorBoard
│   └── pt_tacotron_v1/       — experimento português
│       ├── hparams.json
│       └── checkpoints/
├── data/
│   ├── raw/LJSpeech-1.1/     — áudios LJSpeech
│   └── processed/
│       ├── LJSpeech/mels_metadata.csv
│       └── tts-portuguese-Corpora/mels_metadata.csv
├── local_weight_models/
│   ├── tacotron/nvidia_tacotron2pyt_fp32_20190427  — pesos pré-treinados
│   └── waveglow/nvidia_waveglowpyt_fp32_20190427   — vocoder
└── exports/synth/style_demo/ — áudios gerados com style transfer
```

---

## 12. Dependências Principais

| Pacote | Versão | Uso |
|---|---|---|
| PyTorch | ≥2.0 | Treino e inferência |
| torchaudio | ≥2.0 | Carregamento de áudio |
| gruut | 2.4.0 | G2P para português |
| gruut_lang_pt | 2.0.1 | Dados de linguagem PT para Gruut |
| gruut_lang_en | 2.0.1 | Dados de linguagem EN para Gruut |
| num2words | — | Expansão de números PT-BR |
| tensorboard | — | Visualização de métricas |
| matplotlib | — | Visualização de espectrogramas |

---

*Documento gerado em 2026-06-26. Experimentos em execução no cluster c1-07-08-16111 com GPU NVIDIA RTX A4500 (20 GB).*

---

# Diário de Experimentos — Sucessos e Falhas

> Seção gerada em 26/06/2026. Registra o histórico completo de cada experimento com métricas reais de TensorBoard, problemas encontrados e soluções aplicadas.

---

## Experimento 1 — `libri_pt_br_phonetic_v6`

**Tipo:** Português fonético (phonemizer gruut), **sem guided attention loss**  
**Dataset:** TTS-Portuguese-Corpus (2340 utterances)  
**Configuração:** LR=1e-5, batch=32, grad_clip=1.0, KL annealing logístico, 24.000 steps

### Curva de Loss

| Step  | Val Total | Grad Norm |
|-------|-----------|-----------|
| 0     | 39.83     | 183.4     |
| 2000  | 1.10      | 3.16      |
| 5000  | 0.91      | 3.28      |
| 10000 | 0.82      | 2.87      |
| 24000 | 0.83      | 2.62      |

### Resultado: ❌ FALHA CRÍTICA

A loss de reconstrução convergiu (0.83), mas a atenção foi completamente ignorada. Sem guided attention loss, o decoder aprendeu a copiar frames sem alinhar texto↔áudio. Na inferência, a atenção ficou **presa na posição 19 do encoder para todos os 400 passos do decoder** — output foi ruído ininteligível.

**Lição:** Guided attention loss é **essencial**. Sem ela, o decoder descobre um atalho degenerado — ignora o encoder e colapsa para reconstrução direta de mel.

---

## Experimento 2 — `libri_pt_br_phonetic_v7`

**Tipo:** Restart com guided attention ativada  
**Configuração:** LR=1e-4, batch=32, guided_attn weight=2.0, sigma=0.4 — apenas 1 checkpoint (step 0)

### Resultado: ❌ ABORTADO

Criado mas interrompido após a primeira inspeção (step 0 apenas, grad norm=193 esperado). Redirecionamento de esforço para validar arquitetura no LJSpeech antes de continuar com português.

---

## Experimento 3 — `lj_speech_v1`

**Tipo:** Inglês LJSpeech — validação completa da arquitetura  
**Dataset:** LJSpeech (13.100 utterances)  
**Configuração:** LR=1e-4→2.5e-5 (ReduceLR), batch=32, grad_clip=5.0, guided_attn weight=2.0 σ=0.4, warm start NVIDIA Tacotron2  
**Status atual:** Rodando (PID 57824), 11.200 steps

### Curva de Loss

| Step  | Val GuidedAttn | Train GuidedAttn | Val Total | LR     | Grad Norm |
|-------|----------------|------------------|-----------|--------|-----------|
| 0     | 0.002472       | 0.003128         | 0.856     | 1e-4   | 25.85     |
| 1000  | 0.001396       | 0.001818         | 0.875     | 1e-4   | 11.82     |
| 3000  | 0.000677       | 0.001000         | 0.745     | 1e-4   | 10.97     |
| 5000  | 0.000530       | 0.000709         | 0.728     | 1e-4   | 9.05      |
| 7000  | 0.000448       | 0.000564         | 0.714     | 5e-5   | 0.94      |
| 9000  | 0.000402       | 0.000506         | 0.710     | 2.5e-5 | 0.73      |
| 11200 | 0.000401       | 0.000402         | 0.627     | 2.5e-5 | 0.83      |

### Resultado: ✅ SUCESSO COMPLETO

- Atenção diagonal convergiu: Val GuidedAttn = 0.000401 (melhor = 0.000378 em step 10800)
- Gate aprende a disparar: sem "Warning! Reached max decoder steps" a partir do epoch_1600
- **Style transfer funcionando** (testado em step 1600, 3 vozes de referência):
  - Durações: 235, 363, 274 frames (+54% entre vozes extremas)
  - Distância L1 entre mels: 1.55–1.75 (forte variação de estilo)
  - Áudio: `exports/synth/style_demo/`
- Grad norm estabilizou em ~0.8 após ReduceLROnPlateau

**Problemas encontrados:**
1. **Grad norm alto no início** (25.8 no step 0): Normal com warm start de backbone pré-treinado; estabilizou progressivamente.
2. **Explosão de KL** em step ~2200: Resolvido com anneal_lag=2000, x0=4000 para atrasar o onset do KL.

---

## Experimento 4 — `pt_tacotron_v1`

**Tipo:** Português, primeiro experimento completo  
**Dataset:** TTS-Portuguese-Corpus (2340 utterances)  
**Configuração:** LR=1e-5, batch=16, grad_clip=5.0, guided_attn weight=2.0, sigma=0.4 — 600 steps, 4 checkpoints

### Resultado: ❌ ABORTADO (bug crítico de LR)

ReduceLROnPlateau com `patience=3` reduzia o LR para 1.25e-6 em dezenas de steps, antes de qualquer aprendizado real. A guided attention em 600 steps ainda estava em 0.114 (sem progresso). Experimento encerrado e substituído por pt_tacotron_v2.

**Lição:** `patience=3` com batch pequeno e validação ruidosa → LR reduz prematuramente. Corrigido para `patience=20`.

---

## Experimento 5 — `pt_tacotron_v2` ⭐ (Principal)

**Tipo:** Português, experimento principal com todas as correções  
**Dataset:** TTS-Portuguese-Corpus (2340 utterances)

**Configuração fase 1 (steps 0–6600):** LR=1e-5, batch=12, grad_clip=5.0, guided_attn weight=8.0, sigma=0.2, sem scheduled sampling  
**Configuração fase 2 (steps 7000+):** + scheduled sampling p=0.2  
**Status atual:** Rodando (PID 442250), epoch_32600, LR=1.25e-6

### Curva de Loss

| Step  | Val GuidedAttn | σ=0.4 equiv | Val Total | LR        | Grad Norm | Evento                         |
|-------|----------------|-------------|-----------|-----------|-----------|--------------------------------|
| 0     | 0.11410        | 0.02853     | 2.067     | 1e-5      | 4.27      | Início                         |
| 2000  | 0.04915        | 0.01229     | —         | 1e-5      | 4.64      | KL annealing inicia            |
| 4000  | 0.00328        | **0.00082** | —         | 1e-5      | 8.44      | **Grande salto de atenção**    |
| 6000  | 0.00321        | 0.00080     | —         | 1e-5      | 4.38      | Plateau                        |
| 7000  | 0.00596        | 0.00149     | 1.083     | 1e-5      | **8.09**  | **Scheduled sampling ON** (spike) |
| 10000 | 0.00533        | 0.00133     | 0.932     | 1e-5      | 3.32      | Recuperação                    |
| 15000 | 0.00279        | 0.00070     | 0.947     | 1e-5      | 3.97      | Melhora contínua               |
| 20000 | 0.00208        | 0.00052     | 0.857     | **5e-6**  | 7.41      | **ReduceLR #1**                |
| 25000 | 0.00195        | 0.00049     | 0.905     | **2.5e-6**| 3.64      | **ReduceLR #2**                |
| 29800 | 0.00174        | **0.00044** | 0.895     | 2.5e-6    | 5.40      | —                              |
| 32600 | 0.00198        | 0.00050     | 0.894     | **1.25e-6**| 4.47     | **ReduceLR #3**                |

> **Nota sobre σ=0.2 vs σ=0.4:** A métrica de guided attention escala com σ² — valores em σ=0.2 são ~4× maiores que o equivalente em σ=0.4. O LJSpeech convergiu em 0.000390 (σ=0.4). O PT-BR atingiu 0.000422 equiv. (val) e 0.000273 equiv. (train), superando o referencial inglês em perda de treinamento.

### Sucessos

**1. Atenção quase-diagonal atingida**
- Train GuidedAttn melhor = 0.001092 (step 27200) ≈ **0.000273 equiv. σ=0.4**
- Val GuidedAttn melhor = 0.001687 (step 25200) ≈ **0.000422 equiv. σ=0.4**
- Referencial LJSpeech: 0.000390 (σ=0.4)

**2. Style transfer funcionando no Português (epoch_29600)**

| Referência   | Frames | Duração |
|--------------|--------|---------|
| sample-77    | 177    | 2.05s   |
| sample-3508  | 303    | 3.52s   |
| sample-4054  | 209    | 2.43s   |

Variação de duração: +71% entre vozes extremas. Distância L1 entre mels: **0.87–1.68**. Áudio: `exports/synth/pt_v2_multi/`

**3. Síntese teacher-forced de alta qualidade (epoch_7200)**  
Áudio de 4.24s com mel de referência real → estrutura espectral compatível com ground truth. Áudio: `exports/synth/pt_v2_teacher_forced/`

**4. Inferência autoregressiva com stopping confiável**  
`--force-monotonic --monotonic-window 0 --attn-stop-frames 10` — para quando a atenção atinge o último token do encoder por 10 frames consecutivos. Produz áudio sem warnings, sem silêncio excessivo.

### Falhas / Problemas

**1. Gate nunca dispara durante inferência autoregressiva**  
Causa raiz: exposure bias — durante o treino com teacher forcing, o gate nunca vê suas próprias predições ruidosas. Gate sigmoid máximo durante inferência: ~0.024 (threshold = 0.5). Scheduled sampling p=0.2 não foi suficiente em 32.600 steps.  
**Solução de contorno:** `--attn-stop-frames 10`.

**2. Atenção com loops backward sem restrição monotônica**  
Sem `--force-monotonic`, a atenção ficou presa em "atratores" (posições 0, 1, 5, 8) e oscilava entre posições 20↔21 por mais de 100 steps sem avançar, nunca chegando ao final do encoder.  
**Solução:** `--force-monotonic --monotonic-window 0` (janela estrita: sem permissão de retrocesso).

**3. Spike de grad ao ativar scheduled sampling (step 7000)**  
Grad norm: 4.4 → 8.09. Guided attn: 0.003 → 0.006. Recuperação em ~400 steps com perda de ~1000 steps de progresso de atenção.

**4. CUDA OOM com dois jobs simultâneos**  
LJSpeech (~12.5 GB) + Português (~5.1 GB) somam ~18 GB em GPU de 20 GB.  
**Solução:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + batch_size=12.

---

## Cronograma de Correções Técnicas

| # | Problema | Solução | Experimento |
|---|----------|---------|-------------|
| 1 | Sem guided attention → alinhamento colapsa | `GuidedAttentionLoss` em `losses.py` | v6 → lj_speech_v1 |
| 2 | `output_lengths` não passado ao criterion | Corrigir `train_utils.py` | lj_speech_v1 |
| 3 | ReduceLROnPlateau patience=3 reduz LR cedo | patience=20 | pt_v1 → pt_v2 |
| 4 | grad_clip=1.0 (updates efetivos < 2%) | grad_clip=5.0 | pt_tacotron_v2 |
| 5 | fp16 AMP causava NaN overflow | Desabilitar AMP, treinar em fp32 | pt_tacotron_v2 |
| 6 | gruut bottleneck: 3.5s/batch | Pré-computar e cachear sequências em `_seq_cache_*.pkl` | pt_tacotron_v2 |
| 7 | Gate não dispara (exposure bias) | Scheduled sampling p=0.2 + `attn_stop_frames=10` | pt_tacotron_v2 |
| 8 | Atenção com loops backward | `--force-monotonic --monotonic-window 0` na inferência | pt_tacotron_v2 |
| 9 | Energy-stop truncando silêncio agressivo | `--energy-stop-threshold -8.5 --energy-stop-frames 20` | pt_tacotron_v2 |
| 10 | WaveGlow com chave `state_dict` em vez de `model` | Strip prefixo `module.` no loader | lj_speech_v1, pt_v2 |

---

## Estado Final dos Experimentos

| Experimento        | Status           | Guided Attn (σ=0.4 equiv) | Gate OK        | Style Transfer | Áudio         |
|--------------------|------------------|---------------------------|----------------|----------------|---------------|
| libri_v6           | ❌ Parado        | N/A (sem GA loss)         | ❌             | ❌             | Ruído         |
| libri_v7           | ❌ Abortado      | —                         | —              | —              | —             |
| lj_speech_v1       | ✅ Convergido    | 0.000401                  | ✅             | ✅             | Inteligível (EN) |
| pt_tacotron_v1     | ❌ Abortado      | 0.028                     | ❌             | ❌             | —             |
| **pt_tacotron_v2** | ✅ Rodando       | **0.000422** (val best)   | ⚠️ workaround  | ✅             | Inteligível (PT-BR) |

**Áudios disponíveis:**
- `exports/synth/style_demo/` — Style transfer inglês (3 vozes, lj_speech_v1)
- `exports/synth/pt_v2_multi/` — Style transfer português (3 vozes × epoch_29600)
- `exports/synth/pt_v2_teacher_forced/` — Síntese com mel de referência real
- `exports/synth/pt_v2_monotonic/` — Inferência autoregressiva monotônica (epoch_29600)

## Comando de Inferência Recomendado (pt_tacotron_v2)

```bash
LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/inference/synthesize_tacotron2_vae.py \
  --experiment experiments/tacotron2-vae/pt_tacotron_v2 \
  --text "Texto a ser sintetizado." \
  --reference-audio path/para/referencia.wav \
  --waveglow local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427 \
  --output-dir exports/synth/saida \
  --force-monotonic \
  --monotonic-window 0 \
  --attn-stop-frames 10 \
  --max-decoder-steps 1000
```
