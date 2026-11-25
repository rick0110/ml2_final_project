# R11 — Curva de N na receita original, 3 seeds: +0,27 bit por duplicação até 128, achata em 241

- **Data:** 23/11/2026 (cluster)
- **Caminho:** [01 — lei de escala](../caminhos/01_lei_de_escala/README.md)
- **Veredito:** confirmado com 3 seeds por ponto

## Pergunta

Com 3 seeds por ponto e a avaliação em 38 vozes, os bits de voz crescem de forma regular com N? Até onde?

## Montagem

Receita original ([protocolo](../protocolo.md)), K = 32, 2.500 passos, seeds 1–3, N ∈ {4, 8, 16, 32, 64, 128, 241}. 241 é o máximo do LibriTTS train-clean-100 com K = 32.

## Resultado

| N | Seeds | Bits de voz | Copiadas (%) | WER (%) |
|---|---|---|---|---|
| 4 | 1 | 0,51 | 35 | 14,9 |
| 8 | 3 | 0,70 ± 0,01 | 57 ± 1 | 13,6 ± 0,9 |
| 16 | 3 | 1,10 ± 0,06 | 65 ± 6 | 12,1 ± 1,1 |
| 32 | 3 | 1,23 ± 0,02 | 70 ± 1 | 11,3 ± 0,3 |
| 64 | 3 | 1,66 ± 0,01 | 77 ± 4 | 13,4 ± 0,9 |
| 128 | 3 | 1,75 ± 0,04 | 79 ± 1 | 11,1 ± 0,3 |
| 241 | 3 | 1,76 ± 0,08 | 81 ± 1 | 11,9 ± 0,6 |

Ajuste linear em log2 N (todas as seeds): **0,27 ± 0,03 bit por duplicação** de N = 8 a 128; 0,22 ± 0,04 incluindo 241.

## Conclusão

- O crescimento é regular e as seeds concordam muito bem (desvio de 0,01–0,08 bit), bem melhor que a medida binária da fase do Colab.
- A inclinação confirma a estimativa de [R10](R10_bits_gargalo_do_decodificador.md) (0,31 [0,17; 0,47]) com intervalo muito menor.
- **De 128 para 241 não há ganho** nesta receita. Depois se viu que boa parte desse limite era da receita de treino ([R15](R15_rampa_do_kl.md), [R16](R16_curva_de_N_sem_kl.md)).

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant base --K 32 --step 2500`.
