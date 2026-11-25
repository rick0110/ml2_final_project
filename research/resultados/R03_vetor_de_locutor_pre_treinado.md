# R03 — Vetor de locutor pré-treinado: ignorado pelo treino (negativo)

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [05 — representação da voz](../caminhos/05_representacao_da_voz/README.md)
- **Veredito:** negativo

## Pergunta

Um codificador de locutor treinado em milhares de vozes (ECAPA-TDNN, congelado) dá ao modelo uma descrição melhor de vozes novas?

## Montagem

Segunda entrada de estilo: o vetor ECAPA, somado ao texto codificado ao lado do vetor do VAE (`-spk`). Três treinos, seed 1, 2.500 passos, avaliação antiga.

## Resultado

| N × K | Variante | WER treinadas | WER novas | Copiadas: treinadas | Copiadas: novas |
|---|---|---|---|---|---|
| 8 × 8 | normal | 21% | 14% | 45% | 26% |
| 8 × 8 | com vetor de locutor | 18% | 14% | 43% | 30% |
| 8 × 128 | normal | 11% | 8% | 91% | 51% |
| 8 × 128 | com vetor de locutor | 13% | 8% | 88% | 46% |
| 2 × 128 | normal | 13% | 8% | 70% | 4% |
| 2 × 128 | com vetor de locutor | 14% | 8% | 70% | 4% |

Com o VAE zerado e só o vetor de locutor, nenhuma voz é copiada (0% nos três treinos).

## Conclusão

O modelo não usa a entrada nova: o VAE vê o áudio alvo no treino e já entrega a voz, então o treino não tem motivo para usar o vetor. Na medida de bits ([R10](R10_bits_gargalo_do_decodificador.md)), o ECAPA sozinho transmite 0,00 bit.

## Dados

README, seção 9.
