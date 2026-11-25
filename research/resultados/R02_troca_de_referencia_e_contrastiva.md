# R02 — Troca de referência e perda contrastiva não reduzem a necessidade de dados (negativo)

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [05 — representação da voz](../caminhos/05_representacao_da_voz/README.md)
- **Veredito:** negativo

## Pergunta

O problema é o código de estilo carregar o conteúdo da frase? Se for, tirar o estilo de outra frase do mesmo locutor, ou forçar códigos parecidos para o mesmo locutor, deveria ajudar.

## Montagem

Três treinos repetidos de duas formas: estilo tirado de outra frase do mesmo locutor (`-swap`) e, além disso, perda contrastiva no latente com peso 0,5 (`-contrast`). 2.500 passos, estilo a 100%, avaliação antiga (4 vozes novas).

## Resultado

| N × K | Variante | WER treinadas | WER novas | Copiadas: treinadas | Copiadas: novas |
|---|---|---|---|---|---|
| 8 × 8 | normal | 21% / 26% | 14% / 24% | 45% / 44% | 26% / 39% |
| 8 × 8 | troca | 23% / 25% | 21% / 24% | 52% / 48% | 27% / 32% |
| 8 × 8 | troca + contrastiva | 27% / 23% | 20% / 21% | 42% / 48% | 27% / 31% |
| 8 × 128 | normal (seed 1) | 11% | 8% | 91% | 51% |
| 8 × 128 | troca (seed 1) | 12% | 8% | 87% | 38% |
| 8 × 128 | troca + contrastiva (seed 1) | 14% | 10% | 85% | 39% |

## Conclusão

Nada mudou além da variação entre seeds. Com estilo a 60%, a contrastiva troca cópia por WER sem melhorar o compromisso. Impedir o latente de carregar a frase não reduz a quantidade de dados necessária: o gargalo parece estar em quantas vozes e frases o decodificador viu.

## Ressalvas

Peso e temperatura da contrastiva não foram ajustados; poucas seeds.

## Dados

README, seção 9; `reports/coverage/results/`.
