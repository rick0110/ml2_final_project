# R18 — Duração do treino sem KL: ganhos cada vez menores depois de 10.000 passos

- **Data:** 25/11/2026 (cluster)
- **Caminho:** [03 — receita de treino](../caminhos/03_receita_de_treino/README.md)
- **Veredito:** indício (uma seed a 20.000 passos)

## Pergunta

Sem a rampa do KL, até onde vale a pena treinar?

## Montagem

`-nokl`, N = 128, K = 32. Os runs mais longos continuam dos checkpoints dos mais curtos (a taxa de aprendizado é constante depois do aquecimento).

## Resultado

| Passos | Seeds | Bits de voz | Copiadas (%) | WER (%) |
|---|---|---|---|---|
| 2.500 | 3 | 1,81 ± 0,06 | 80 | 12,0 |
| 5.000 | 3 | 2,44 ± 0,05 | 88 | 8,3 |
| 10.000 | 2 | 2,84 ± 0,13 | 91 | 6,8 |
| 20.000 | 1 | 2,96 | 92 | 6,0 |

## Conclusão

Cada dobra de passos rende menos: +0,63, +0,40, +0,12 bit. Com N = 128 e K = 32 o treino está perto de saturar por volta de 10.000–20.000 passos. Como mais locutores pedem mais treino ([R16](R16_curva_de_N_sem_kl.md)), a comparação justa entre N deveria usar treinos perto da saturação de cada N. Em curso: 20.000 passos em N = 241.

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant nokl --N 128 --K 32`.
