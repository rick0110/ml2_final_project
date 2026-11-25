# R13 — Latente de 8 e 2 dimensões: a maior parte da perda vem de reinicializar as cabeças do VAE

- **Data:** 23–24/11/2026 (cluster)
- **Caminho:** [05 — representação da voz](../caminhos/05_representacao_da_voz/README.md)
- **Veredito:** a previsão "latente de 8 não muda nada" falhou, mas o controle mostra que a comparação estava confundida

## Pergunta

Quantas dimensões o latente de estilo precisa? Previsão registrada (seção 8.10 da nota teórica): com 8 dimensões, cópia e bits inalterados; com 2, caem.

## Montagem

`-z8`, `-z2`: latente menor, com as cabeças do VAE (fc1–fc3) reinicializadas ao carregar o checkpoint de 32 dimensões. Controle `-zr32`: latente de 32 com as mesmas cabeças reinicializadas. Receita original, K = 32, 2.500 passos.

## Resultado

| Variante | N | Seeds | Bits de voz | Copiadas (%) | WER (%) |
|---|---|---|---|---|---|
| original (32, sem reinicializar) | 128 | 3 | 1,75 ± 0,04 | 79 ± 1 | 11,1 ± 0,3 |
| `-zr32` (32, reinicializado) | 128 | 3 | 0,97 ± 0,06 | 72 ± 3 | 14,6 ± 0,6 |
| `-z8` | 128 | 3 | 0,57 ± 0,05 | 63 ± 6 | 15,4 ± 2,1 |
| `-z2` | 128 | 2 | 0,06 ± 0,01 | 51 ± 12 | 23,2 ± 0,8 |
| `-z8` | 8 | 3 | 0,45 ± 0,02 | 49 ± 4 | 17,1 ± 1,7 |
| `-z8` | 32 | 3 | 0,62 ± 0,07 | 57 ± 9 | 16,0 ± 1,8 |
| `-z8` | 241 | 1 | 0,42 | 72 | 13,2 |

## Conclusão

- **Reinicializar as cabeças custa 0,78 bit** (1,75 → 0,97) em 2.500 passos: o modelo não recupera o latente a tempo.
- **O tamanho em si custa 0,40 bit** (zr32 0,97 → z8 0,57, mesma reinicialização). O latente de 2 dimensões zera os bits (0,06) e dobra o WER.
- Com z8 os bits nem crescem com N (0,45 → 0,62 → 0,57 → 0,42): o latente pequeno vira o gargalo.
- A previsão "z8 não muda" está errada mesmo descontando a reinicialização, mas a comparação justa exigiria treinar mais tempo (sem a rampa do KL, [R15](R15_rampa_do_kl.md)).

## Dados

[`dados/execucoes_cluster.csv`](dados/execucoes_cluster.csv); `python scripts/coverage/results_table.py --variant base z8 z2 zr32 --step 2500 --K 32`.
