# R14 — Metade das frases da voz padrão: +4 pontos de cópia, bits iguais

- **Data:** 23–24/11/2026 (cluster)
- **Caminho:** [06 — atração pela voz padrão](../caminhos/06_atracao_pela_voz_padrao/README.md)
- **Veredito:** previsão de cópia confirmada no limite inferior; os bits não mudam, o que indica efeito da medida, não da voz

## Pergunta

A voz padrão (2.000 frases do LJSpeech) "puxa" as saídas para ela? Previsão registrada (seção 8.11 da nota teórica): com metade das frases (M = 1.000), de 4 a 9 pontos a menos de vozes não copiadas em N = 128; separar as teorias exige cerca de 6 treinos por condição.

## Montagem

`-lj1000`: 1.000 frases da voz padrão, mesmas frases de teste. N = 128, K = 32, 2.500 passos, 8 seeds.

## Resultado

| Variante | Seeds | Bits de voz | Copiadas (%) | WER (%) |
|---|---|---|---|---|
| original (2.000 frases) | 3 | 1,75 ± 0,04 | 79 ± 1 | 11,1 ± 0,3 |
| `-lj1000` (1.000 frases) | 8 | 1,83 ± 0,08 | 83 ± 2 | 10,0 ± 0,8 |

## Conclusão

- A cópia sobe 4 pontos, no limite inferior da previsão.
- Os bits sobem só 0,08 (dentro do desvio entre seeds). A medida "copiada" compara com a voz padrão; com menos voz padrão no treino a saída se afasta dela e a medida melhora, sem que o decodificador alcance mais vozes.
- Reforça usar os bits como métrica principal.

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant base lj1000 --N 128 --K 32 --step 2500`.
