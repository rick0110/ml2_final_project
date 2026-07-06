# Tacotron2-VAE — Síntese de fala com prosódia interpretável

**Projeto final de ML2 (IMPA Tech)** · [Página do projeto](https://rick0110.github.io/ml2_final_project) · [Relatório (PDF)](reports/relatorio_final.pdf)

Sistema de síntese de fala (*Text-to-Speech*) baseado no **Tacotron 2** com um **VAE** para modelagem de prosódia. A contribuição central do trabalho é substituir o *pooling* usual do *reference encoder* (último estado da GRU) por um ***pooling* por atenção espectral**, tornando o condicionamento de estilo interpretável — além de produzir um modelo especificamente para o português a partir do corpus emocional VERBO.

---

## 1. O problema e a contribuição

No Tacotron2-VAE, um *reference encoder* comprime um áudio de referência em um vetor latente `z` que captura características globais de estilo (ritmo, tom, velocidade) e condiciona a síntese. Na prática esse latente frequentemente "morre": o modelo aprende a reconstruir o espectrograma ignorando `z` (*posterior collapse*).

Este trabalho:

1. **Implementa métodos contra o colapso do posterior** — *free bits* (0,5 nats/dim) e *KL annealing* cíclico;
2. **Troca o *pooling* do *reference encoder*** — em vez de usar só o último estado da GRU, uma atenção aprende pesos αₜ sobre todos os blocos da varredura do espectrograma (bandas de canais mel, dos graves aos agudos):

   ```
   sₜ = wᵀhₜ + b        α = softmax(s)        e = Σ αₜ hₜ
   ```

   O custo é desprezível (E/2 + 1 parâmetros) e os pesos αₜ formam um **mapa de atenção legível sobre o espectro**;
3. **Adiciona a perda de atenção guiada** (Tachibana et al. 2017) sobre a atenção *location-sensitive* do decoder — ausente no artigo de referência (Zhang et al. 2019) — para resolver o desalinhamento severo do início do treino.

## 2. Resultados obtidos

Três modelos treinados: *baseline* (último estado da GRU) e variante com atenção no **LJSpeech** (inglês), e a variante em português no **VERBO** (locutora f1).

| | LJ *baseline* | LJ atenção | VERBO f1 |
|---|---|---|---|
| Iterações | 11,2 mil | 35,5 mil | 9,5 mil |
| Parâmetros | 28,9M | 28,9M | 7,7M |
| Latente z | 32 | 32 | 16 |
| Dimensões ativas do latente | **0/32** (var. máx. 0,004) | **32/32** | **15/16** |

- **Colapso do posterior:** o *baseline* reconstrói um pouco melhor (0,596 vs. 0,722) mas às custas de um latente morto; com o *pooling* por atenção o latente permanece informativo.
- **Transferência de estilo (LJSpeech):** condicionado a três referências de prosódia contrastante (grave/lenta, expressiva, rápida), o *baseline* produz áudios praticamente idênticos; o modelo com atenção muda ritmo e F0 de forma audível e consistente com a referência.
- **Interpretabilidade da atenção:** os pesos αₜ concentram-se em bandas específicas do espectro. No VERBO, a **entropia normalizada** da atenção correlaciona-se com a emoção da referência: alegria 0,99, raiva 0,99 e surpresa 0,98 (atenção quase uniforme — energia em banda larga) contra tristeza 0,67, neutro 0,70 e desgosto 0,76 (atenção concentrada em poucas bandas). A correlação com a energia do bloco é fraca (|r| ≤ 0,38): a atenção não é um mero detector de volume.
- **Transferência de emoção em português:** condicionando a síntese às 7 referências emocionais do VERBO, ritmo e registro mudam consistentemente com a emoção. Limitações: ~4,7 min de dados e vocoder treinado em inglês geram artefatos — o objetivo é estudar o espaço de estilo, não produção.
- **Manipulação direta do latente:** mover uma única dimensão influente a ±3σ produz mudanças graduais e audíveis de ritmo/registro; no *baseline*, ±3σ de uma dimensão morta não muda nada.
- **Interpolação esférica:** em alta dimensão a massa do prior gaussiano concentra-se numa casca de raio √L. A *lerp* atravessa o interior da casca (áudio perde definição harmônica em t = 0,25–0,50); a *slerp* caminha pelo arco e preserva a qualidade. No *baseline* (θ = 14°) os dois caminhos coincidem — o caso degenerado.

Áudios, figuras e análises completas: [rick0110.github.io/ml2_final_project](https://rick0110.github.io/ml2_final_project).

## 3. Arquitetura

```
Texto ──► Encoder (embedding + 3×Conv1d + BiLSTM) ──►(+)──► Atenção ──► Decoder ──► Postnet ──► WaveGlow ──► áudio
                                                      ▲    location-    autoregressivo
Mel de referência ──► Reference encoder (VAE) ──► e ──┘    sensitive    + gate de parada
                      6×Conv2d (1ª CoordConv) + GRU
                      + pooling por atenção → μ, σ → z
```

- **Text encoder:** embedding de símbolos → 3 convoluções 1D (k=5) → BiLSTM.
- **Reference encoder:** 6 convoluções 2D de *stride* 2 (a primeira é uma CoordConv) reduzem o mel 80×T; uma GRU percorre o resultado em T′ = T/64 blocos (bandas do espectro) e o *pooling* por atenção resume os estados em um vetor, projetado em (μ, log σ²); amostra-se `z`, convertido em vetor de estilo **e** somado a todas as posições da memória do encoder.
- **Decoder:** atenção *location-sensitive* (conteúdo + posição via convolução sobre os αₜ acumulados), prenet, 2 LSTMs, projeção para o quadro mel e para o *gate* de parada; postnet residual de 5 convoluções.
- **Vocoder:** WaveGlow 256ch pré-treinado pela NVIDIA (congelado, σ = 0,6).

## 4. Perdas de treinamento

```
L = L_mel + L_gate + β(passo)·L_KL + λ·L_GA
```

| Perda | Descrição |
|---|---|
| `L_mel` | MSE do mel (decoder + postnet), mascarada no *padding* |
| `L_gate` | Entropia cruzada binária do sinal de parada |
| `L_KL` | KL com *free bits* de 0,5 nats/dimensão |
| `β(passo)` | *Annealing* cíclico: rampa 0 → 0,2 a cada 4.000 passos, após 2.000 de espera |
| `L_GA` | Atenção guiada: penaliza massa de atenção longe da diagonal (LJ: λ=2,0, σ=0,4; VERBO: λ=50, σ=0,2) |

## 5. Dados

| Corpus | Falas | Duração | Idioma | Entrada de texto |
|---|---|---|---|---|
| LJSpeech 1.1 | 13.100 | ~23,9 h | inglês | caracteres (`english_cleaners`) |
| VERBO (locutora f1) | 98 | ~4,7 min | português, 7 emoções | fonemas IPA (`gruut`) |

Áudio: 22.050 Hz, 80 canais mel, FFT 1024, *hop* 256, f_max 8 kHz.

## 6. Como treinar

```bash
# 1. Pré-processar o corpus (exemplo: VERBO)
python scripts/preprocess/preprocess_Verbo.py \
  --input-dir data/raw/VERBO-Dataset \
  --out-dir data/processed/VERBO/mels

# 2. Treinar (receita do modelo VERBO do relatório — rede reduzida)
python src/training/training-tacotron2-vae/train.py \
  --batch-size 16 --learning-rate 1e-4 --grad-clip-thresh 1.0 \
  --anneal-function cyclical \
  --guided-attention-weight 50.0 --guided-attention-sigma 0.2 \
  --text-cleaner portuguese_phonetic_cleaners \
  --artifacts-dir data/processed/VERBO \
  --symbols-embedding-dim 256 --encoder-embedding-dim 256 --style-dim 256 \
  --z-latent-dim 16 --decoder-rnn-dim 512 --attention-rnn-dim 512 \
  --prenet-dim 128 --postnet-embedding-dim 256 \
  --experiment-name meu_experimento
```

O modelo do relatório usa apenas a locutora f1 do VERBO. Para os modelos LJSpeech, use `--text-cleaner english_cleaners` e as dimensões cheias (defaults de `hparams.py`).

### TensorBoard

```bash
tensorboard --logdir experiments/<experimento>/logs
```

Loga perdas (treino/val/teste, com KL e atenção guiada separadas), espectrograma alvo vs. predito por inferência, valores singulares do latente (diagnóstico de colapso) e normas de gradiente por camada.

## 7. Estrutura do repositório

```text
ml2_final_project/
├── data/
│   ├── raw/                       # Corpora originais (LJSpeech, VERBO, ...)
│   └── processed/                 # Tensores .pt + manifestos mels_metadata.csv
├── src/
│   ├── data/                      # Datasets e DataLoaders (loader_vae_tacotron, loader_waveglow)
│   ├── models/
│   │   ├── tacotron2_vae/         # Encoder, decoder, atenção, VAE-GST, hparams
│   │   └── waveglow/              # Vocoder (usado congelado, pré-treinado pela NVIDIA)
│   └── training/
│       └── training-tacotron2-vae/  # train.py, perdas, loop de treino, TensorBoard
├── scripts/
│   ├── preprocess/                # Extração de mel-espectrogramas por corpus
│   ├── inference/                 # Utilitários de síntese
│   └── site_assets/               # Geração das figuras/áudios da página do projeto
├── notebooks/                     # Demos e análises (ver abaixo)
├── experiments/                   # Checkpoints, hparams.json, logs TensorBoard
├── docs/                          # Página do projeto (GitHub Pages)
└── reports/                       # Relatório final (LaTeX + PDF)
```

## 8. Notebooks

| Notebook | O que faz |
|---|---|
| `tacotron2_vae_demo.ipynb` | Loop de treino ao vivo no VERBO com dashboard (mel alvo vs. gerado, perdas, PCA do latente) |
| `attention_analysis.ipynb` | Análise dos pesos de atenção do *reference encoder* por emoção |
| `attention_deep_analysis.ipynb` | Análise aprofundada da atenção do *reference encoder* |
| `style_transfer_demo.ipynb` | Transferência de emoção ponta a ponta: 7 referências do VERBO → síntese + playback |
| `demo_spectral_analisys.ipynb` | Estatísticas espectrais comparativas VERBO vs. LJSpeech (PCA/t-SNE) |
| `demo_audio.ipynb` | Comparação de vocoders por cópia-síntese: Griffin-Lim vs. WaveGlow |
| `demo_vocoder_waveglow.ipynb` | Loop de treino/fine-tuning do WaveGlow (exploratório; o projeto usa o checkpoint NVIDIA congelado) |
| `testing_nvidia_tarcotron2.ipynb` | Sanidade: Tacotron2 + WaveGlow oficiais da NVIDIA via `torch.hub` |

## 9. Instalação

```bash
git clone https://github.com/rick0110/ml2_final_project.git
cd ml2_final_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash initialize_directories.sh
```

Requisitos: Python 3.10+, GPU CUDA para treino.

## Créditos

**Discentes:** Beatriz Nunes, Herivelton Siqueira, Richard Viana · **Professores:** Francisco Ganacim, Daniel Silveira Yukimura Lopez · **IMPA Tech**, julho de 2026.

**Referências principais:** Hsu et al. 2019 (Tacotron2-VAE) · Zhang et al. 2019 (VAE-TTS) · Shen et al. 2018 (Tacotron 2) · Skerry-Ryan et al. 2018 (*reference encoder*) · Prenger et al. 2019 (WaveGlow) · Tachibana et al. 2017 (atenção guiada) · White 2016 (*slerp*) · Torres Neto et al. 2018 (corpus VERBO).
