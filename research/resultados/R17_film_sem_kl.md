# R17 — FiLM sem KL: 3,08 bits com 241 locutores, o melhor resultado até agora

- **Data:** 24–25/11/2026 (cluster)
- **Caminho:** [02 — gargalo do decodificador](../caminhos/02_gargalo_do_decodificador/README.md)
- **Veredito:** promissor; poucas seeds a 10.000 passos

## Pergunta

O ganho do FiLM ([R12](R12_film.md)) se soma ao de tirar a rampa do KL ([R15](R15_rampa_do_kl.md))?

## Montagem

`-filmnokl`, K = 32, 5.000 e 10.000 passos.

## Resultado

| N | Sem KL, 5.000 | FiLM sem KL, 5.000 | Sem KL, 10.000 | FiLM sem KL, 10.000 |
|---|---|---|---|---|
| 8 | 0,89 ± 0,03 (3) | 1,10 ± 0,01 (2) | 1,07 ± 0,01 (2) | 1,20 (1) |
| 16 | 1,56 ± 0,06 (3) | 1,73 (1) | 1,72 ± 0,06 (3) | 1,74 (1) |
| 32 | 1,69 ± 0,06 (3) | 2,00 ± 0,00 (2) | 2,11 ± 0,10 (2) | 2,11 (1) |
| 64 | 2,14 ± 0,06 (3) | 2,39 (1) | 2,38 ± 0,07 (3) | 2,53 (1) |
| 128 | 2,44 ± 0,05 (3) | 2,63 ± 0,05 (3) | 2,84 ± 0,13 (2) | 2,67 (1) |
| 241 | 2,53 ± 0,06 (3) | 2,98 ± 0,08 (2) | 2,90 (1) | **3,08** (1) |

Bits de voz (seeds entre parênteses). FiLM sem KL a 10.000 passos, N = 241: 97% das amostras copiadas, WER 5,9%.

## Conclusão

- A 5.000 passos o FiLM soma +0,17 a +0,45 bit em todos os N, com o maior ganho em 241.
- A 10.000 passos a vantagem diminui e fica dentro do ruído na maioria dos pontos (uma seed). O FiLM parece acelerar o aprendizado mais do que elevar o teto; isso ainda precisa de seeds.
- 3,08 bits é cerca de 60% do teto de 5,25 e de 65–70% do que a fala real carrega (4,2–4,7 bits, [R10](R10_bits_gargalo_do_decodificador.md)).

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant nokl filmnokl --K 32`.
