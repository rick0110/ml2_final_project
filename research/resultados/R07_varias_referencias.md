# R07 — Média do código de várias referências: ajuda nas 4 vozes novas, não nas 38 externas

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [05 — representação da voz](../caminhos/05_representacao_da_voz/README.md)
- **Veredito:** negativo para a cópia; positivo só como estabilizador (WER menor)

## Pergunta

O piso de vozes novas não copiadas é ruído de estimar a voz a partir de uma frase só? Se for, usar a média do código de estilo de R frases do mesmo locutor deveria ajudar.

## Montagem

Sem treino novo: na avaliação, código = média dos códigos de R = 1, 4 ou 16 frases (`--n-refs` em `scripts/eval/synth_check.py`). Medida por locutor (`sim_spk > sim_default`).

## Resultado

Avaliação antiga (4 vozes novas), R = 1 → 4:

| Treino | Treinadas | Novas | Externas (4) |
|---|---|---|---|
| 128 × 8 | 66% → 66% | 47% → 67% | 77% → 83% |
| 128 × 32, seed 1 | 84% → 71% | 74% → 78% | 95% → 94% |
| 128 × 32, seed 2 | 73% → 68% | 59% → 69% | 74% → 97% |
| 200 × 32 | 77% → 68% | 57% → 75% | 76% → 90% |

Avaliação larga (38 vozes externas), por locutor, R = 1 → 4: 8 × 32: 66% → 64%; 32 × 32: 81% → 79%; 128 × 32: 88% → 84%. WER: 13% → 12%, 12% → 10%, 12% → 9%.

## Conclusão

Nas 38 vozes a média de 4 referências não sobe a cópia (−2,5 pontos, IC [−6,4; +0,8]) e muda os bits de voz entre −0,40 e +0,31 sem direção. O ganho visto nas 4 vozes novas não se generaliza. O que a média faz de consistente é baixar o WER. Junto com [R03](R03_vetor_de_locutor_pre_treinado.md)/[R04](R04_dropout_de_estilo.md): o gargalo não é descrever a voz nova, é o decodificador conseguir renderizá-la.

## Dados

Seções 8.2, 8.4, 8.6, 8.8 e 8.10 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).
