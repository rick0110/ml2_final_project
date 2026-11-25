# R16 — Curva de N sem KL: cerca de +0,4 bit por duplicação, ainda achatando de 128 para 241

- **Data:** 24–25/11/2026 (cluster)
- **Caminho:** [01 — lei de escala](../caminhos/01_lei_de_escala/README.md)
- **Veredito:** confirmado (2–3 seeds por ponto, exceto 241 a 10.000 passos)

## Pergunta

Sem a rampa do KL ([R15](R15_rampa_do_kl.md)) e com treinos mais longos, a curva de bits contra N fica mais inclinada? O achatamento em 241 some?

## Montagem

`-nokl`, K = 32, 5.000 e 10.000 passos, N ∈ {8, 16, 32, 64, 128, 241}.

## Resultado

| N | Original, 2.500 (3 seeds) | Sem KL, 5.000 | Sem KL, 10.000 |
|---|---|---|---|
| 8 | 0,70 ± 0,01 | 0,89 ± 0,03 (3) | 1,07 ± 0,01 (2) |
| 16 | 1,10 ± 0,06 | 1,56 ± 0,06 (3) | 1,72 ± 0,06 (3) |
| 32 | 1,23 ± 0,02 | 1,69 ± 0,06 (3) | 2,11 ± 0,10 (2) |
| 64 | 1,66 ± 0,01 | 2,14 ± 0,06 (3) | 2,38 ± 0,07 (3) |
| 128 | 1,75 ± 0,04 | 2,44 ± 0,05 (3) | 2,84 ± 0,13 (2) |
| 241 | 1,76 ± 0,08 | 2,53 ± 0,06 (3) | 2,90 (1) |

Bits de voz (seeds entre parênteses). Copiadas a 10.000 passos: de 67% (N = 8) a 95% (N = 241). WER: de 8,8% a 6,0%.

Inclinação em log2 N (todas as seeds):

| Receita | N de 8 a 128 | N de 8 a 241 |
|---|---|---|
| original, 2.500 | 0,27 ± 0,03 | 0,22 ± 0,04 |
| sem KL, 5.000 | 0,37 ± 0,05 | 0,33 ± 0,04 |
| sem KL, 10.000 | 0,41 ± 0,06 | 0,38 ± 0,05 |

## Conclusão

- Treinar mais tempo sem KL sobe a curva inteira e a deixa mais inclinada: o ganho por duplicação de N vai de 0,27 para 0,41 bit.
- Mais locutores pedem mais treino: o ganho de 5.000 para 10.000 passos é maior com N grande (+0,40 em 128) do que com N pequeno (+0,18 em 8).
- O achatamento de 128 para 241 diminui, mas continua (+0,06 a 10.000 passos, com uma seed em 241). Hipóteses em aberto: falta de passos, falta de frases por locutor (K = 32) ou limite real; em teste no [caminho 01](../caminhos/01_lei_de_escala/README.md) e no [caminho 04](../caminhos/04_locutores_vs_frases/README.md).

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant nokl --K 32`.
