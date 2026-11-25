# FiLM sem KL: mais seeds e 20.000 passos

**Início:** 25/11/2026 · **Caminho:** [02](../README.md)

## Objetivo

Em [R17](../../../resultados/R17_film_sem_kl.md) o FiLM soma +0,17 a +0,45 bit a 5.000 passos, mas a 10.000 a vantagem quase some (uma seed). Duas leituras:

- **o FiLM eleva o teto:** a vantagem volta a aparecer com mais seeds e com 20.000 passos;
- **o FiLM só acelera:** os dois convergem para o mesmo valor.

## Runs (na fila ou rodando)

| Run | Passos | Para quê |
|---|---|---|
| `n128_k32_s2-filmnokl`, `n241_k32_s2-filmnokl` | 10.000 (continuam de 5.000) | seed 2 a 10.000 |
| `n16_k32_s2/s3-filmnokl`, `n64_k32_s2/s3-filmnokl` | 5.000 e 10.000 | seeds dos pontos intermediários |
| `n241_k32_s1-filmnokl` | 20.000 (continua de 10.000) | teto em 241 |
| `n16_k32_s1/s2/s3-film`, `n64_k32_s1/s2/s3-film` | 2.500 (receita original) | completar a curva do FiLM de R12 |

## Previsões registradas antes dos resultados (25/11/2026)

- **Critério principal:** compara-se `n241_k32_s1-filmnokl` com `n241_k32_s1-nokl` a 20.000 passos.
  - Se a diferença for **≥ 0,2 bit**, o FiLM eleva o teto.
  - Se for **< 0,1 bit**, só acelera o aprendizado.
  - Entre 0,1 e 0,2 bit, espera mais seeds.
- **A 10.000 passos, com 2 seeds:** a vantagem média do FiLM em N ≥ 64 fica entre 0 e +0,2 bit.
- **Na receita original, a 2.500 passos:** o FiLM em N = 64 fica entre 1,8 e 2,0 bits (original: 1,66).
