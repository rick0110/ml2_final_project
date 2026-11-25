# R04 — Dropout de estilo: o vetor de locutor passa a ser usado, mas não reduz a necessidade de locutores (negativo)

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [05 — representação da voz](../caminhos/05_representacao_da_voz/README.md)
- **Veredito:** negativo

## Pergunta

Se forçarmos a voz a passar pelo vetor de locutor pré-treinado (zerando o VAE em metade dos itens), as vozes novas melhoram?

## Montagem

`-spkdrop`: vetor ECAPA + VAE com dropout de estilo 0,5. Três treinos, uma seed, 2.500 passos, avaliação antiga. "Só vetor" = VAE zerado na avaliação; "vetor + VAE" = estilo a 100%.

## Resultado

| N × K | Variante | WER treinadas | WER novas | Copiadas: treinadas | Copiadas: novas |
|---|---|---|---|---|---|
| 8 × 128 | normal | 11% | 8% | 91% | 51% |
| 8 × 128 | dropout, só vetor | 9% | 6% | 71% | 5% |
| 8 × 128 | dropout, vetor + VAE | 14% | 8% | 88% | 37% |
| 32 × 8 | normal (2 seeds) | 17% / 17% | 13% / 9% | 58% / 51% | 47% / 47% |
| 32 × 8 | dropout, só vetor | 11% | 7% | 9% | 1% |
| 32 × 8 | dropout, vetor + VAE | 22% | 15% | 57% | 60% |
| 32 × 128 | normal (2 seeds) | 12% / 10% | 5% / 5% | 86% / 86% | 61% / 57% |
| 32 × 128 | dropout, só vetor | 9% | 6% | 64% | 47% |
| 32 × 128 | dropout, vetor + VAE | 12% | 6% | 83% | 62% |

## Conclusão

Com 8 locutores o vetor sozinho decora as vozes treinadas (71%) e quase não copia as novas (5%). Com 32 locutores e 128 frases ele já copia 47% das novas sozinho, mas o conjunto fica igual ao treino normal. Um codificador treinado em milhares de vozes não dispensa o decodificador de ver muitos locutores: o limite é quantas vozes o decodificador ouviu, seja qual for o caminho por onde a voz chega.

## Dados

README, seção 9.
