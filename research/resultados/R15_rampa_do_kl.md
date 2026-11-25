# R15 — A rampa do KL limitava o treino: com ela, treinar mais piora; sem ela, melhora muito

- **Data:** 24/11/2026 (cluster)
- **Caminho:** [03 — receita de treino](../caminhos/03_receita_de_treino/README.md)
- **Veredito:** confirmado com 3 seeds

## Pergunta

Por que treinar 5.000 passos em vez de 2.500 piorava os bits de voz (1,75 → 1,42 em N = 128)?

## Montagem

Na receita original o peso do KL fica em 0 até o passo 2.000 e depois sobe 0,05 a cada 1.000 passos. Controle `-nokl`: peso do KL sempre 0. K = 32.

Evolução do KL na receita original (run `n241_k32_s1`):

| Passo | Peso do KL | KL (nats) |
|---|---|---|
| 2.000 | 0 | 3,40 |
| 2.500 | 0,025 | 1,56 |
| 3.000 | 0,05 | 1,07 |
| 4.000 | 0,10 | 0,72 |
| 4.975 | 0,15 | 0,58 |

## Resultado

| Receita | N | 2.500 passos | 5.000 passos | 10.000 passos |
|---|---|---|---|---|
| original | 128 | 1,75 ± 0,04 (3) | 1,42 (1) | — |
| original | 241 | 1,76 ± 0,08 (3) | 1,43 ± 0,07 (3) | — |
| FiLM | 128 | 2,06 ± 0,02 (3) | 1,64 (1) | — |
| sem KL | 128 | 1,81 ± 0,06 (3) | 2,44 ± 0,05 (3) | 2,84 ± 0,13 (2) |
| sem KL | 241 | — | 2,53 ± 0,06 (3) | 2,90 (1) |

Bits de voz (seeds entre parênteses). O WER cai com o treino nas duas receitas: na original, em N = 241, de 11,9% para 8,2%; sem KL, em N = 128, de 12,0% (2.500) para 8,3% (5.000) e 6,8% (10.000).

## Conclusão

- A rampa do KL comprime o latente (o KL cai de 3,4 para 0,6 nats entre os passos 2.000 e 5.000) e tira dele a informação da voz: com o KL subindo, treinar mais **perde** cerca de 0,33 bit.
- Sem a rampa, treinar mais tempo **ganha**: +0,63 bit de 2.500 para 5.000 passos e mais +0,40 até 10.000, em N = 128.
- Até o passo 2.500 as duas receitas são quase iguais (1,75 contra 1,81), como esperado: a rampa só começa no passo 2.000.
- Consequência: todos os experimentos seguintes usam `-nokl` e treinos mais longos ([R16](R16_curva_de_N_sem_kl.md)–[R18](R18_duracao_do_treino.md)).

## Ressalvas

Sem KL o VAE deixa de ser regularizado; amostrar estilos do prior deixa de fazer sentido (não é usado na clonagem, que sempre parte de uma referência).

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `metrics.jsonl` de cada run nas máquinas e no Drive.
