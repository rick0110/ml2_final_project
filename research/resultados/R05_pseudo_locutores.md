# R05 — Pseudo-locutores por mudança de velocidade não contam como locutores (negativo, previsto)

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [07 — aumento de vozes](../caminhos/07_aumento_de_vozes/README.md)
- **Veredito:** negativo; a previsão "não contam" acertou os dois números

## Pergunta

Criar vozes novas alterando a velocidade das gravações (tom e formantes mudam juntos) aumenta o N efetivo de graça?

## Montagem

`-aug` (`scripts/coverage/augment_voices.py`): cada frase de treino ganha uma cópia com velocidade alterada entre 1,5 e 4 semitons (41 e 54 pseudo-locutores). Mesmos passos (2.500), avaliação antiga. Previsões registradas antes, na seção 5 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).

## Resultado

| Treino | Copiadas: treinadas | Copiadas: novas | WER novas | Previsto se não contam | Previsto se contam |
|---|---|---|---|---|---|
| 8 × 8 normal (2 seeds) | 45% / 44% | 26% / 39% | 14% / 24% | | |
| 8 × 8 com pseudo-locutores | 50% | **28%** | 14% | ~31% | ~50% |
| 8 × 128 normal (2 seeds) | 91% / 88% | 51% / 37% | 8% / 9% | | |
| 8 × 128 com pseudo-locutores | 84% | **42%** | 8% | ~44% | ~55% |

## Conclusão

O N que conta é o número de vozes **reais**. Mudar a velocidade desloca a voz numa direção só do espaço de vozes; as vozes novas diferem em muitas (timbre, fonte glotal, articulação, ritmo). Sobra aumentar vozes com um conversor de voz treinado em muitos locutores, que ficou bloqueado ([caminho 07](../caminhos/07_aumento_de_vozes/README.md)).

## Dados

Seção 6 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).
