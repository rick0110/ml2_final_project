# R06 — 128 e 200 locutores na avaliação antiga: a lei acerta K = 8, o resto é ruído de seed

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [01 — lei de escala](../caminhos/01_lei_de_escala/README.md)
- **Veredito:** previsão arriscada parcialmente confirmada; a avaliação com 4 vozes novas mostrou-se ruidosa demais (motivou [R08](R08_avaliacao_larga_38_vozes.md))

## Pergunta

A lei ajustada na grade N × K prevê quase nenhum ganho acima de 32 locutores (56–57% de vozes novas copiadas com 128). Isso vale?

## Montagem

`n128_k8_s1`, `n128_k32_s1`, réplica `n128_k32_s2` e `n200_k32_s1`. 2.500 passos, estilo a 100%. Previsões nas seções 8.1 e 8.5 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).

## Resultado

| Treino | Treinadas copiadas (previsto → medido) | Novas copiadas (previsto → medido) | Externas (4 vozes) | WER novas |
|---|---|---|---|---|
| 128 × 8 | 52% → 52% | 56% → 55% | 86% | 8% |
| 128 × 32, seed 1 | 82% → 76% | 57% → 74% | 86% | 8% |
| 128 × 32, seed 2 | — → 66% | 60% (réplica) → 56% | 86% | 8% |
| 200 × 32, seed 1 | — → 66% | 60–68% → 64% | 91% | 7% |

## Conclusão

- Com K = 8 a lei acertou a menos de 1 ponto.
- Os 74% de 128 × 32 eram em boa parte sorte da seed: a réplica deu 56% (média 65%).
- De 128 para 200 locutores não houve ganho mensurável **nesta avaliação**. Duas leis (queda em 1/N ou em 1/√N) ajustam os 16 treinos igualmente bem.
- O que limitava a conclusão era a medida: 4 vozes novas × 25 frases dão ruído entre seeds de até 18 pontos. Isso motivou a avaliação em 38 vozes ([R08](R08_avaliacao_larga_38_vozes.md)), que mostrou ganho real de 32 para 128.

## Dados

Seções 8.1–8.6 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).
