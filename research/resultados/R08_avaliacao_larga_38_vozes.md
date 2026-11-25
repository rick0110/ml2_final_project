# R08 — Avaliação em 38 vozes externas: o ganho de 32 para 128 locutores é real

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [01 — lei de escala](../caminhos/01_lei_de_escala/README.md)
- **Veredito:** confirmado; a avaliação em 38 vozes passou a ser o padrão

## Pergunta

Com uma avaliação menos ruidosa (38 locutores do LibriTTS test-clean, nenhum ouvido por nenhum modelo), o ganho com N continua acima de 32 locutores?

## Montagem

Três treinos com K = 32 (`-wide`, N = 8, 32, 128), uma seed, 2.500 passos, receita original. 950 amostras por treino (38 vozes × 25 frases); intervalos por reamostragem dos locutores (10.000 vezes), diferenças pareadas nas mesmas vozes. Previsões na seção 8.7 da nota teórica.

## Resultado

| Treino | Externas copiadas (IC 95%) | WER externas |
|---|---|---|
| 8 × 32 | 57% (43–70) | 13% |
| 32 × 32 | 70% (58–81) | 12% |
| 128 × 32 | 79% (68–89) | 12% |

## Conclusão

- Ganho de 32 para 128: 9 pontos (IC 95% de 1,5 a 17), contra 13 pontos de 8 para 32. Desfaz a leitura de que acima de 128 não há ganho, que vinha das 4 vozes.
- O WER não muda com N: mais locutores melhoram a cópia sem custo de inteligibilidade.
- A cópia é quase tudo-ou-nada por voz (há locutores com 0% e com 100% em todos os treinos).
- A revisão estatística posterior (seção 8.10) mostrou que três pontos de uma seed não fixam o expoente nem escolhem a lei. Isso motivou as 3 seeds por ponto no cluster ([R11](R11_curva_de_N_receita_original.md)).

## Dados

Seções 8.7, 8.8 e 8.10 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md); `reports/teoria/04_dados.md`.
