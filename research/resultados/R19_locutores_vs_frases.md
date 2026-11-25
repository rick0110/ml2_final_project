# R19 — Com a mesma quantidade de dados, mais locutores rendem mais bits que mais frases por locutor

- **Data:** 24–25/11/2026 (cluster)
- **Caminho:** [04 — locutores vs frases](../caminhos/04_locutores_vs_frases/README.md)
- **Veredito:** confirmado nas duas receitas; várias células com uma seed

## Pergunta

Com o mesmo total de frases dos locutores do LibriTTS, é melhor ter muitos locutores com poucas frases ou poucos locutores com muitas?

## Montagem

Pares N × K com o mesmo total N·K. Receita original (2.500 passos) e sem KL (5.000 passos).

## Resultado

| Total de frases | N × K | Original, 2.500 | Sem KL, 5.000 |
|---|---|---|---|
| 1.024 | 8 × 128 | 0,74 (1) | 0,94 (1) |
| 1.024 | 32 × 32 | 1,23 ± 0,02 (3) | 1,69 ± 0,06 (3) |
| 1.024 | 128 × 8 | 1,36 ± 0,08 (3) | 1,93 (1) |
| 4.096 | 32 × 128 | 1,38 (1) | 1,84 ± 0,03 (2) |
| 4.096 | 128 × 32 | 1,75 ± 0,04 (3) | 2,44 ± 0,05 (3) |
| 1.928 / 7.712 | 241 × 8 / 241 × 32 | — | 2,16 (1) / 2,53 ± 0,06 (3) |

Bits de voz (seeds entre parênteses).

## Conclusão

- Com 1.024 frases, passar de 8 para 128 locutores quase dobra os bits (0,74 → 1,36; 0,94 → 1,93). Com 4.096, de 32 para 128 locutores dá +0,37 a +0,60 bit.
- Frases por locutor também ajudam com N fixo (128 × 8 → 128 × 32: +0,39 e +0,51 bit), mas menos que o mesmo aumento de dados em locutores.
- Confirma, em bits e com as 38 vozes, a leitura da grade antiga ([R01](R01_grade_N_x_K.md)): para vozes novas, o que mais conta é o número de vozes.

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --K 8 128` e `--N 128 --K 32`.
