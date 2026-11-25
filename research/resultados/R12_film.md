# R12 — FiLM no decodificador: o ganho cresce com N (até +0,44 bit em 241)

- **Data:** 23–24/11/2026 (cluster)
- **Caminho:** [02 — gargalo do decodificador](../caminhos/02_gargalo_do_decodificador/README.md)
- **Veredito:** apoio parcial: a previsão registrada (≥ +0,5 bit em N = 8 e em N = 128) não foi atingida, mas o ganho é real e cresce com N

## Pergunta

Se o gargalo é o ponto onde o estilo entra no decodificador, deixar o vetor de estilo modular os estados das LSTMs a cada passo (FiLM) aumenta os bits de voz?

## Montagem

`-film` ([protocolo](../protocolo.md)): FiLM inicializado em zero (o modelo começa idêntico ao original). Receita original, K = 32, 2.500 passos. Previsão registrada antes (seção 8.12 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md)): C sobe pelo menos 0,5 bit em N = 8 e em N = 128; se ficar dentro do intervalo do treino normal, o ponto de entrada do estilo não é o gargalo.

## Resultado

| N | Original: bits (seeds) | FiLM: bits (seeds) | Diferença | FiLM: copiadas | FiLM: WER |
|---|---|---|---|---|---|
| 8 | 0,70 ± 0,01 (3) | 0,71 ± 0,07 (3) | +0,01 | 53% | 10,9% |
| 16 | 1,10 ± 0,06 (3) | 1,22 (1) | +0,12 | 71% | 13,4% |
| 32 | 1,23 ± 0,02 (3) | 1,45 ± 0,01 (3) | +0,22 | 75% | 12,4% |
| 128 | 1,75 ± 0,04 (3) | 2,06 ± 0,02 (3) | +0,31 | 85% | 11,3% |
| 241 | 1,76 ± 0,08 (3) | 2,20 ± 0,05 (3) | +0,44 | 88% | 10,7% |

Inclinação: 0,33 ± 0,03 bit por duplicação (8–128), contra 0,27 ± 0,03 na receita original.

## Conclusão

- Em N = 8 o FiLM não muda nada; o ganho aparece e cresce com N, e em 241 o FiLM continua subindo onde a receita original estaciona.
- Leitura: com poucas vozes, o limite é a cobertura (o decodificador viu poucas vozes); com muitas, o modo de condicionar passa a limitar, e o FiLM ajuda.
- Sem custo de inteligibilidade (WER igual ou menor).

## Ressalvas

N = 16 com uma seed. FiLM só nas LSTMs do decodificador (não na pós-rede nem na atenção).

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant base film --step 2500 --K 32`.
