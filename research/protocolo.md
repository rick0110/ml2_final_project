# Protocolo experimental

Como os treinos do experimento de cobertura são montados e avaliados. Vale para todos os resultados a partir de R11 (cluster). Os resultados R01–R10 (fase do Colab) usam a mesma montagem, com as diferenças indicadas em cada arquivo.

## Montagem de um treino

- **Ponto de partida:** o modelo em inglês já treinado (Tacotron 2 + VAE, a partir do Tacotron 2 pré-treinado da NVIDIA), sempre o mesmo checkpoint (`ckpt/en_final`).
- **Dados de treino:** 2.000 frases do LJSpeech (a "voz padrão", uma locutora) + **N locutores do LibriTTS train-clean-100 com K frases cada**. As seleções são aninhadas: os locutores de N = 8 estão dentro de N = 32, as frases de K = 8 dentro de K = 32 (`scripts/coverage/coverage_manifest.py`). O máximo de locutores disponível é 241.
- **Treino:** batch 48, taxa de aprendizado 1e-4 com 500 passos de aquecimento, checkpoint a cada 500 passos, 2.500 passos por padrão.
- **Rampa do KL (receita original):** o peso do KL fica em 0 até o passo 2.000 e depois sobe 0,05 a cada 1.000 passos (0,15 no passo 5.000). A variante `-nokl` desliga a rampa.
- **Seeds:** o número depois de `_s` no nome do run muda a ordem dos dados e a inicialização do que é novo.

## Nome dos runs

`n{N}_k{K}_s{seed}[-variante]`, por exemplo `n128_k32_s2-nokl`. Variantes (definidas em `scripts/coverage/colab_job.sh`):

| Sufixo | O que muda |
|---|---|
| (nenhum) | receita original |
| `-film` | o vetor de estilo também modula, por FiLM, os estados das LSTMs do decodificador a cada passo (inicializado em zero) |
| `-nokl` | sem rampa do KL (peso do KL sempre 0) |
| `-filmnokl` | FiLM e sem rampa do KL |
| `-z8`, `-z2` | latente de 8 ou 2 dimensões (cabeças do VAE reinicializadas) |
| `-zr32` | latente de 32 com as cabeças do VAE reinicializadas (controle de `-z8`/`-z2`) |
| `-lj1000` | 1.000 frases da voz padrão em vez de 2.000 |
| `-swap`, `-contrast`, `-spk`, `-spkdrop`, `-aug`, `-vcaug` | variantes da fase do Colab (R02–R05) |

O número de passos vai na fila de jobs, não no nome: o mesmo run pode ser avaliado em 5.000 e 10.000 passos. Pedir mais passos a um run que já existe continua do último checkpoint (a taxa de aprendizado é constante depois do aquecimento, então continuar equivale a treinar direto).

## Avaliação

Para cada checkpoint avaliado:

- **25 frases de teste** fora do treino.
- **Referências:** 8 vozes treinadas, 4 vozes novas do mesmo corpus e **38 vozes externas** (LibriTTS test-clean, que nenhum modelo ouviu). As conclusões usam as 38 externas: com 4 vozes, o ruído entre seeds chegava a 18 pontos.
- **Síntese** com o estilo inteiro (força 1,0) e vocoder HiFi-GAN universal.

## Métricas

- **Bits de voz** (`bits_ce`, a métrica principal): informação mútua entre a voz do áudio gerado e a identidade da referência, C = I(voz da saída; referência), estimada pela cota de entropia cruzada a partir da identificação entre as 38 vozes externas (`scripts/coverage/voice_bits.py`). Teto: log2 38 = 5,25 bits. Também se calcula a cota de Fano (`bits_fano`), mais conservadora, e H(Y) e H(Y|S).
- **Copiadas:** parcela das amostras cujo áudio é mais parecido com a referência do que com a voz padrão (`sim_ref > sim_default`, codificador de locutor Resemblyzer). É fácil de ler, mas depende da distância de cada voz à voz padrão (ver [caminho 06](caminhos/06_atracao_pela_voz_padrao/README.md)).
- **WER:** taxa de palavras erradas, transcrevendo o áudio gerado com o Whisper. Mede a inteligibilidade.

## Como gerar as tabelas

```
python scripts/coverage/results_table.py                         # tudo
python scripts/coverage/results_table.py --variant nokl --step 10000
```

Os dados vêm de `research/resultados/dados/execucoes_cluster.csv`: uma linha por avaliação (run, variante, N, K, seed, passos, bits, cópia, WER, máquina).
