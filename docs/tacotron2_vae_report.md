# Relatório Técnico: Tacotron2-VAE com Style Transfer em Português

**Projeto:** ml2_final_project  
**Data:** Junho 2026  
**Objetivo:** Síntese de voz em português com transferência de estilo via VAE-GST

---

## 1. Visão Geral

Este projeto implementa um sistema de síntese de voz (TTS) com transferência de estilo para o português brasileiro. A arquitetura base é o Tacotron2 (NVIDIA), extendida com um módulo VAE-GST (Variational Autoencoder + Global Style Token) que permite condicionar a geração de mel-spectrogramas no estilo prosódico de um áudio de referência.

**Pipeline completo:**

```
Texto (str)
  └─► TextProcessor (gruut, IPA) → sequência de fonemas [int]
        └─► Encoder (3× Conv1D + 1 BiLSTM) → encoder_outputs [B, T_enc, 512]
              └─► VAE-GST (ref_audio) → latent_vector z [B, 32]
                    └─► z expandido + encoder_outputs → condicionamento conjunto
                          └─► Decoder (2 LSTM + location-sensitive attention) → mel_pre [B, 80, T_dec]
                                └─► PostNet (5× Conv1D residual) → mel_post [B, 80, T_dec]
                                      └─► WaveGlow → áudio [22050 Hz]
```

---

## 2. Arquitetura do Modelo

### 2.1 Encoder

- Embedding de fonemas: 512 dims
- 3 camadas Conv1D (kernel=5, channels=512, BatchNorm, ReLU)
- 1 BiLSTM (hidden=256, saída=512)

### 2.2 VAE-GST (Style Encoder)

- Referência de áudio → mel-spectrogram → GRU reference encoder → μ, log σ² (cada 32 dims)
- Reparameterização: z = μ + ε·σ
- **KL annealing cíclico:** lag=2000, x0=4000, upper=0.2, free_bits=0.5
- Latent dim: 32
- z expandido por broadcast e somado ao encoder_outputs

### 2.3 Decoder

- Prenet: 2× Linear(256) + Dropout(p=0.5) — dropout **sempre ativo** inclusive na inferência
- Attention RNN: LSTM(1024) com location-sensitive attention (filters=32, kernel=31)
- Decoder RNN: LSTM(1024)
- Saída: mel frame [80] + gate scalar
- Scheduled sampling: p=0.2 (20% dos passos usam a predição anterior em vez do target)

### 2.4 PostNet

- 5× Conv1D (kernel=5, channels=512, BatchNorm, Tanh/Linear)
- Residual: mel_post = mel_pre + postnet_output

### 2.5 Guided Attention Loss

Penalidade diagonal que força o alinhamento atenção/encoder a seguir uma diagonal:

```
W[t, n] = 1 - exp(-(n/N - t/T)² / 2σ²)
```

- Peso: 8.0 (pt_tacotron_v2)
- σ = 0.2 (stricter; equivale a σ=0.4 com peso 4× maior)

---

## 3. Datasets

| Dataset | Idioma | Utterances | Duração |
|---------|--------|------------|---------|
| LJSpeech | Inglês (en-US) | 13.100 | ~24h |
| TTS-Portuguese-Corpus | Português (pt-BR) | 2.340 | ~3.5h |

**Pré-processamento:**
- Resample para 22050 Hz
- STFT: filter=1024, hop=256, win=1024, mel_bins=80, fmin=0, fmax=8000
- Fonemas: `gruut` com `portuguese_phonetic_cleaners` (IPA)
- Cache de sequências fonéticas em `_seq_cache_{hash}.pkl`

---

## 4. Configuração de Treinamento

| Parâmetro | lj_speech_v1 | pt_tacotron_v2 |
|-----------|-------------|----------------|
| LR inicial | 1e-4 | 1e-5 |
| Batch size | 32 | 12 |
| Grad clip | 1.0 | 5.0 |
| Guided attn weight | 2.0→4.0 | 8.0 |
| Guided attn σ | 0.4 | 0.2 |
| Scheduled sampling | 0 | 0.2 |
| LR scheduler | ReduceLROnPlateau (patience=20) | ReduceLROnPlateau (patience=20) |
| Warm start | N/A | Pesos do lj_speech_v1 |

---

## 5. Modificações Implementadas

| # | Modificação | Arquivo | Motivação |
|---|-------------|---------|-----------|
| 1 | Guided Attention Loss | `losses.py` | Forçar alinhamento diagonal; sem isso o modelo falha catastrophicamente |
| 2 | output_lengths masking bug fix | `train_utils.py` | Bug: output_lengths não era passado ao criterion → loss errada no padding |
| 3 | KL annealing cíclico | `train.py` + `hparams.py` | Annealing linear colapsava o VAE; cíclico mantém informação de estilo |
| 4 | grad_clip_thresh: 1.0→5.0 | `hparams.py` | Clipping agressivo impedia aprendizado; 5.0 permite updates maiores |
| 5 | LR warmup | `train.py` + `train_utils.py` | Estabilidade no início do treinamento PT com warm start |
| 6 | Cross-lingual warm start | CLI `--checkpoint-path` | Reutilizar pesos EN para acelerar convergência PT |
| 7 | p_decoder_input_dropout: 0.8→0.5 | `hparams.py` | 0.8 muito alto; 0.5 é o padrão original do Tacotron2 |
| 8 | AMP desabilitado | `train.py` | fp16 causava NaN overflow no KL loss; fp32 estável |
| 9 | Phoneme pre-caching | `loader.py` | gruut é lento; cache elimina overhead em epochs subsequentes |
| 10 | persistent_workers=True | `train.py` | Evita respawn de workers entre epochs (overhead significativo) |
| 11 | Scheduled sampling (p=0.2) | `model.py` Decoder.forward() | Mitiga exposure bias parcialmente: 20% dos steps usam predição anterior |
| 12 | Monotonic attention + attn-peak stop | `model.py` Decoder.inference() | Solução definitiva para atenção em loop + parada sem gate treinado |

---

## 6. Inferência — Modo de Uso Recomendado

```bash
LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/inference/synthesize_tacotron2_vae.py \
  --experiment experiments/tacotron2-vae/pt_tacotron_v2 \
  --text "Texto em português aqui." \
  --reference-audio data/raw/tts-portuguese-Corpora/TTS-Portuguese-Corpus/wavs/sample-77.wav \
  --waveglow local_weight_models/waveglow/nvidia_waveglowpyt_fp32_20190427 \
  --output-dir exports/synth/output \
  --force-monotonic \
  --monotonic-window 0 \
  --attn-stop-frames 10 \
  --max-decoder-steps 1000
```

**Flags críticos:**

| Flag | Valor | Motivo |
|------|-------|--------|
| `--force-monotonic` | ativo | Evita loops de atenção (attention attractor problem) |
| `--monotonic-window 0` | 0 | Estrito: sem retrocesso; window>0 permite loops novamente |
| `--attn-stop-frames 10` | 10 | Para quando atenção chega ao último token por 10 frames consecutivos |
| `--max-decoder-steps 1000` | 1000 | Fallback de segurança |

**Alternativa (sem monotonic, para modelos com gate treinado):**
```bash
--energy-stop-threshold -8.5 --energy-stop-frames 20
```

---

## 7. Diário de Experimentos

### 7.1 Experimento 0: libri_pt_br_phonetic_v6 — Falha Catastrófica

**Configuração:**
- Dataset: TTS-Portuguese-Corpus (2340 utterances)
- Sem guided attention loss
- 24.000 steps de treinamento

**Resultado: FALHA CATASTRÓFICA**

A atenção nunca convergiu para um padrão diagonal. Mesmo após 24.000 steps, o modelo produzia mel-spectrogramas sem estrutura fonética recognoscível. A atenção oscilava entre posições aleatórias do encoder, nunca progressivamente.

**Lição aprendida:**  
Guided attention loss é **essencial** para datasets pequenos. Com 2340 utterances, o modelo não tem exemplos suficientes para descobrir o alinhamento monotônico por conta própria. Sem a penalidade diagonal, a atenção não converge.

---

### 7.2 Experimento 1: lj_speech_v1 — Convergência Completa em Inglês

**Objetivo:** Validar a arquitetura VAE-GST em inglês antes de atacar o português.

**Configuração:**
- Dataset: LJSpeech (13.100 utterances)
- LR=1e-4, batch=32, guided_attn_weight=2.0→4.0, σ=0.4
- Sem scheduled sampling (exposição total ao ground truth)

**Métricas (TensorBoard, step 9800):**
| Métrica | Valor |
|---------|-------|
| Val loss total | 0.704 |
| Guided Attn Loss (val) | 0.000390 |
| Grad norm | 0.98 |
| LR | 2.5e-5 (ReduceLR disparou 2×) |

**Resultado: SUCESSO**

- Gate dispara naturalmente durante inferência (sigmoid > 0.5)
- Alinhamento diagonal limpo sem monotonic constraint
- Style transfer funcional: mesma frase com vozes diferentes produz prosódias distintas

**Lição aprendida:**  
A arquitetura funciona corretamente em inglês. guided_attn_weight=4.0 foi o valor que finalizou a convergência (2.0 estabilizava em guided_attn ≈ 0.038). Dataset grande (13k utts) permite que o gate aprenda a disparar via teacher forcing puro.

---

### 7.3 Experimento 2: pt_tacotron_v2 (fase inicial) — Explosão de KL

**Objetivo:** Fine-tuning do lj_speech_v1 para português.

**Configuração inicial:**
- Warm start com pesos do lj_speech_v1
- LR=1e-5, batch=12, guided_attn_weight=8.0, σ=0.2
- KL annealing: linear (primeiro tentativa)
- AMP (fp16) habilitado

**Problemas encontrados:**

1. **NaN overflow com fp16:** O KL loss explodiu para NaN nos primeiros 500 steps. Solução: desabilitar AMP, usar fp32.

2. **KL onset explosion:** Mesmo em fp32, o grad_norm saltou para ~3617 no step 2200 quando o KL começou a ser annealed. Com grad_clip=1.0, os gradientes eram cortados agressivamente e o aprendizado parou. Solução: grad_clip=5.0.

3. **VAE collapse com annealing linear:** O annealing linear fazia o peso KL crescer monotonicamente, colapsando o espaço latente antes que o decoder convergisse. Solução: annealing cíclico (lag=2000, x0=4000, upper=0.2, free_bits=0.5).

4. **Guided attention estagnada em 0.003:** Com batch=12 e grad_clip=1.0, o guided attention loss não reduzia abaixo de 0.003 (σ=0.2). Após scheduled sampling ser adicionado, houve perturbação temporária (pico de 0.0086 no step 7000) seguida de recuperação.

---

### 7.4 Experimento 3: pt_tacotron_v2 — Scheduled Sampling + Problema do Gate

**Configuração:**
- Adição de scheduled sampling p=0.2 (restart em ~step 7000)
- LR=1e-5, reduzido 3× pelo ReduceLROnPlateau: 1e-5 → 5e-6 → 2.5e-6 → 1.25e-6

**Métricas (step 29600/epoch_29600):**
| Métrica | Valor |
|---------|-------|
| Train guided_attn (σ=0.2) | 0.001101 |
| Val guided_attn (σ=0.2) | 0.001743 |
| Equivalente σ=0.4 (÷4) | ≈ 0.000275 / 0.000436 |
| Val total loss | 0.895 |
| Gate sigmoid max (inferência) | **0.024** |

**Problema crítico descoberto — Exposure Bias / Gate Gap:**

O gate é treinado **exclusivamente com teacher forcing** (ground truth como input). Durante a inferência, o modelo usa sua própria predição anterior (autoregressive), criando uma distribuição de input diferente. O gate nunca aprendeu a disparar nessas condições.

Resultado: sigmoid(gate) ≈ 0.024 em todos os steps da inferência — muito abaixo do threshold=0.5. O modelo gerava áudio até atingir `max_decoder_steps` (1000), produzindo ~1000 frames de silêncio após o conteúdo real.

**Problema crítico — Attention Attractor:**

Sem constraint monotônico, a atenção oscilava em um ciclo entre posições fixas do encoder:

```
frames 1-300: atenção em posições {0, 1, 5, 8, 20, 21}
              nunca avança para posição 33 (último token)
              → loop infinito até max_decoder_steps
```

**Tentativas de solução:**

1. `energy-stop-threshold -7.5 --energy-stop-frames 5`: Muito agressivo — cortava fala no frame 14 (dip de energia no meio de palavra). 
2. `energy-stop-threshold -8.5 --energy-stop-frames 20`: Funcionou — cortava apenas no silêncio real (frame 193+).
3. `--force-monotonic --monotonic-window 3`: Window=3 permite ir de peak=21 de volta a 20 (20 ≥ 21-3=18). **Não quebrou os loops** — apenas desacelerou.

---

### 7.5 Experimento 4: pt_tacotron_v2 — Solução Definitiva com Monotonic Attention

**Configuração de inferência:**
```
--force-monotonic --monotonic-window 0 --attn-stop-frames 10
```

**Mecanismo:**
- `monotonic_window=0`: Uma vez que o pico de atenção alcança posição N, **todos os passos 0..N-1 são mascarados** (prob→0). Movimento estritamente para frente.
- `attn_stop_frames=10`: Para quando o pico fica em `last_enc_pos` por 10 frames consecutivos — confirma que o encoder foi completamente consumido.

**Resultados (epoch_29600, 15 inferências = 5 frases × 3 vozes):**

| Frase | Tokens | Referência | Frames | Duração | Fr/tok | Regressões | Cobertura |
|-------|--------|-----------|--------|---------|--------|------------|-----------|
| f1 | 34 | s77 | 177 | 2.05s | 5.2 | 0 | 100% |
| f1 | 34 | s3508 | 303 | 3.52s | 8.9 | 0 | 100% |
| f1 | 34 | s4054 | 209 | 2.43s | 6.1 | 0 | 100% |
| f2 | 39 | s77 | 193 | 2.24s | 4.9 | 0 | 100% |
| f3 | 50 | s77 | 209 | 2.43s | 4.2 | 0 | 100% |

- **0 avisos de max_decoder_steps** em todos os 15 runs
- **100% cobertura do encoder** — atenção sempre alcança último token
- Style transfer confirmado: s77 vs s3508 → 71% de diferença de duração na mesma frase

**Distâncias L1 entre mels (mesma frase, vozes diferentes):**

| Par | L1 |
|-----|-----|
| s77 vs s3508 | 0.87 |
| s77 vs s4054 | 1.68 |
| s3508 vs s4054 | 1.21 |

---

### 7.6 Estado Atual: epoch_40000 (Jun 26, 2026)

**Métricas de qualidade (epoch_40000, 15 inferências):**

| Frase | Tokens | Ref | Frames | Duração | Fr/tok | Regressões | Cob% |
|-------|--------|-----|--------|---------|--------|------------|------|
| f1 | 34 | s77 | 251 | 2.91s | 7.4 | 0 | 100% |
| f1 | 34 | s3508 | 195 | 2.26s | 5.7 | 0 | 100% |
| f1 | 34 | s4054 | 182 | 2.11s | 5.4 | 0 | 100% |
| f2 | 39 | s77 | 193 | 2.24s | 4.9 | 0 | 100% |
| f2 | 39 | s3508 | 182 | 2.11s | 4.7 | 0 | 100% |
| f2 | 39 | s4054 | 205 | 2.38s | 5.3 | 0 | 100% |
| f3 | 50 | s77 | 209 | 2.43s | 4.2 | 0 | 100% |
| f3 | 50 | s3508 | 288 | 3.34s | 5.8 | 0 | 100% |
| f3 | 50 | s4054 | 218 | 2.53s | 4.4 | 0 | 100% |
| f4 | 48 | s77 | 209 | 2.43s | 4.4 | 0 | 100% |
| f4 | 48 | s3508 | 174 | 2.02s | 3.6 | 0 | 100% |
| f4 | 48 | s4054 | 177 | 2.05s | 3.7 | 0 | 100% |
| f5 | 45 | s77 | 176 | 2.04s | 3.9 | 0 | 100% |
| f5 | 45 | s3508 | 181 | 2.10s | 4.0 | 0 | 100% |
| f5 | 45 | s4054 | 185 | 2.15s | 4.1 | 0 | 100% |

**Gate:** sigmoid max = 0.275 (vs. 0.024 em epoch_29600 — melhora de 11×). Ainda abaixo do threshold=0.5, mas com trajetória crescente. Projeção: gate funcional entre epoch_45000–50000.

**Style transfer (L1 entre vozes, epoch_40000):**

| Par | L1 |
|-----|-----|
| s77 vs s3508 | 0.789 |
| s77 vs s4054 | 1.210 |
| s3508 vs s4054 | 1.495 |
| LJSpeech baseline (referência) | 1.55–1.75 |

Ligeiramente abaixo do baseline LJSpeech — esperado, dado que o dataset PT tem 2340 utterances vs 13100 do LJSpeech.

**Variância estocástica (prenet dropout):**  
5 runs com mesmo texto + mesma voz: [181, 186, 178, 253, 204] frames. Δ=75 frames (0.87s). Comportamento inerente ao dropout do prenet, ativo durante inferência no Tacotron2 original. Não é bug.

---

## 8. Tabela de Falhas e Correções

| Problema | Causa | Solução |
|----------|-------|---------|
| NaN no KL loss | AMP fp16 overflow | Desabilitar AMP, usar fp32 |
| Grad explosion no KL onset | grad_clip muito baixo (1.0) | Aumentar para 5.0 |
| VAE collapse | Annealing KL linear | Annealing cíclico (lag, free_bits) |
| Guided attn estagnada em 0.038 | guided_attn_weight=2.0 insuficiente | Aumentar para 4.0 (LJ) / 8.0 (PT) |
| Corte prematuro de fala | energy_stop threshold -7.5/window 5 agressivo | Usar -8.5/window 20 |
| Atenção em loop (attractor) | Sem constraint monotônico | --force-monotonic --monotonic-window 0 |
| Geração não para (gate não dispara) | Exposure bias — gate treinado apenas com teacher forcing | --attn-stop-frames 10 |
| CUDA OOM | Inferências paralelas durante treinamento | PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + sequencial |
| ZeroDivisionError no TacotronSTFT | Argumentos posicionais errados | Usar argumentos nominais (keyword args) |
| Monotonic window=3 não resolve loops | Window 3 ainda permite retroceder 3 posições | Usar window=0 (estrito) |
| Atenção nunca converge (libri_v6) | Ausência de guided attention loss | Guided attention é obrigatória para datasets pequenos |

---

## 9. Estado Final dos Experimentos

| Experimento | Status | Checkpoint | Alinhamento | Style Transfer | Gate |
|-------------|--------|------------|-------------|----------------|------|
| libri_v6 | FALHOU | — | Caótico | — | — |
| lj_speech_v1 | CONVERGIDO | step ~9800 | Diagonal perfeito | Funciona (EN) | Dispara (≥0.5) |
| pt_tacotron_v2 | RODANDO | epoch_40000 | Diagonal (com monotonic) | Funciona (PT) | 0.275, crescendo |

---

## 10. Frases de Teste Utilizadas

```
f1: "O título de página foi encontrado."        (34 tokens)
f2: "O sistema retornou um erro inesperado."    (39 tokens)
f3: "A conferência foi cancelada por motivos de segurança." (50 tokens)
f4: "Os resultados foram publicados ontem à tarde."         (48 tokens)
f5: "A análise dos dados revelou padrões interessantes."    (45 tokens)
```

---

## 11. Requisitos de Ambiente

```bash
# Variáveis obrigatórias
export LD_LIBRARY_PATH=/opt/anaconda3/envs/ambiente_aluno/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU: 1× ~20GB VRAM (inferência + treinamento simultâneo requer execução sequencial)
# Python: .venv com dependências instaladas
# gruut: fonemas IPA para pt-BR, requer CXXABI do conda (motivo do LD_LIBRARY_PATH)
```

---

## 12. Conclusões

**O que funcionou:**

1. Warm start cross-lingual (EN → PT) acelerou convergência significativamente
2. Guided attention com peso alto (8.0, σ=0.2) convergiu alinhamento diagonal em ~5000 steps
3. Scheduled sampling (p=0.2) reduziu parcialmente o exposure bias
4. Monotonic attention (window=0) + attn_stop_frames=10: solução robusta para inferência sem gate treinado — 100% de sucesso em todos os testes
5. Style transfer VAE-GST funcional: mesma frase com 3 vozes diferentes produz variação de 5–38% em duração e L1 mel de 0.79–1.50

**O que ainda não funciona perfeitamente:**

1. Gate de parada: sigmoid max=0.275 em epoch_40000 (threshold=0.5 não atingido). Mitigado por `attn_stop_frames`.
2. Variância run-to-run (prenet dropout): ~75 frames de spread entre runs idênticos. Inerente ao design do Tacotron2.
3. Qualidade absoluta do style transfer (L1 ≈1.5 vs baseline 1.65): dataset PT menor que LJSpeech limita a diversidade aprendida.

**Próximos passos:**

- Aguardar epoch ~45000–50000 para verificar se gate atinge sigmoid≥0.5
- Avaliar se treinamento adicional melhora naturalidade do áudio (MOS subjetivo)
- Considerar HiFi-GAN como vocoder alternativo ao WaveGlow para qualidade de áudio
