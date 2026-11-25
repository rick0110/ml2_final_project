# Dados dos resultados

## `execucoes_cluster.csv`

Uma linha por avaliação feita no cluster (um run avaliado num número de passos). Colunas:

| Coluna | Significado |
|---|---|
| `run` | nome do run (`n{N}_k{K}_s{seed}[-variante]`, ver [protocolo](../../protocolo.md)) |
| `variant` | variante (`base` = receita original) |
| `N`, `K`, `seed` | locutores, frases por locutor, seed |
| `step` | passos de treino do checkpoint avaliado |
| `bits_ce` | bits de voz, cota de entropia cruzada, 38 vozes externas (métrica principal) |
| `bits_fano` | bits de voz, cota de Fano (mais conservadora) |
| `copied` | parcela das amostras externas com `sim_ref > sim_default` |
| `wer` | taxa de palavras erradas (Whisper) nas amostras externas |
| `machine` | máquina do cluster onde o run está (1–10) |

Os arquivos completos de cada avaliação (`results.csv` por amostra, `voice_bits.json` com intervalos, `embeddings.npz`, áudios) e os checkpoints ficam nas máquinas e no Google Drive ([infra](../../infra.md)).

**Atualizar:** o CSV é gerado no PC de controle lendo as máquinas (script `collect_csv.py` em `~/ml2_backup/tools/`, fora do repositório porque usa os endereços). Depois de atualizar, regenere as tabelas dos resultados afetados com `scripts/coverage/results_table.py`.

Última atualização: 25/11/2026, 122 avaliações.

## Fase do Colab

Os resultados R01–R10 vêm de `reports/coverage/results/` e das análises em `reports/teoria/`.
