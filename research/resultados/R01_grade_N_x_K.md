# R01 — Grade N × K: frases decidem as vozes treinadas, locutores decidem as novas

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [01 — lei de escala](../caminhos/01_lei_de_escala/README.md), [04 — locutores vs frases](../caminhos/04_locutores_vs_frases/README.md)
- **Veredito:** estrutura em dois níveis confirmada; números das vozes novas são ruidosos (4 vozes por avaliação)

## Pergunta

Quando uma voz aparece no treino com poucas frases, o modelo consegue ler texto novo com ela? Como isso muda com o número de locutores (N) e de frases por locutor (K)?

## Montagem

Receita original ([protocolo](../protocolo.md)), 2.500 passos, estilo a 100%. Avaliação antiga: 25 frases, 8 vozes treinadas e 4 vozes novas do mesmo corpus (sem as 38 externas). Onde há dois números, são duas seeds.

## Resultado

| N | K | WER treinadas | WER novas | Copiadas: treinadas | Copiadas: novas |
|---|---|---|---|---|---|
| 2 | 128 | 13% / 11% | 8% / 5% | 70% / 74% | 4% / 2% |
| 8 | 8 | 21% / 26% | 14% / 24% | 45% / 44% | 26% / 39% |
| 8 | 32 | 17% | 14% | 74% | 47% |
| 8 | 128 | 11% / 10% | 8% / 9% | 91% / 88% | 51% / 37% |
| 32 | 8 | 17% / 17% | 13% / 9% | 58% / 51% | 47% / 47% |
| 32 | 32 | 13% | 8% | 80% | 52% |
| 32 | 128 | 12% / 10% | 5% / 5% | 86% / 86% | 61% / 57% |

## Conclusão

- **K decide as vozes treinadas:** com 8 locutores, a cópia vai de 45% (K = 8) a 74% (32) e 90% (128).
- **N decide as vozes novas:** com K = 128, de 3% (N = 2) a 44% (8) e 59% (32).
- **Muitos locutores compensam poucas frases:** 32 × 8 (256 frases) copia vozes novas tanto quanto 8 × 128 (1.024 frases).
- **O latente explica o erro:** em 144 referências, quanto mais longe da voz padrão está o código de estilo, mais a voz é copiada (correlação de postos 0,83) e mais palavras saem erradas (0,45).
- Uma lei de amostragem em dois níveis ajusta a grade (não copiadas ≈ a + A/N + B/(N·K) para vozes novas; a' + C/K + A'/N para treinadas), mas a forma da queda não foi identificada depois ([R06](R06_128_e_200_locutores.md), [R08](R08_avaliacao_larga_38_vozes.md)).

## Ressalvas

4 vozes novas por avaliação; entre seeds a cópia de vozes novas varia até 18 pontos. Medida binária com um único codificador de locutor.

## Dados

`reports/coverage/results/` (resultados por amostra), `scripts/coverage/summarize.py`, `scripts/coverage/fit_law.py`. Detalhes nas seções 1–4 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).
