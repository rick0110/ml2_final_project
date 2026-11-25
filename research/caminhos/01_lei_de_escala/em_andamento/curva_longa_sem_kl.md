# Curva longa sem KL: seeds a 10.000 passos e 20.000 passos em 241 locutores

**Início:** 25/11/2026 · **Caminho:** [01](../README.md)

## Objetivo

1. Completar 3 seeds por ponto da curva sem KL a 10.000 passos ([R16](../../../resultados/R16_curva_de_N_sem_kl.md) tem 1–3).
2. Descobrir se o achatamento de 128 para 241 locutores é falta de passos.

## Runs (na fila ou rodando; estado no inventário)

| Run | Passos | Para quê |
|---|---|---|
| `n8_k32_s2-nokl`, `n32_k32_s2-nokl`, `n128_k32_s3-nokl` | 10.000 (continuam de 5.000) | seeds da curva |
| `n241_k32_s2-nokl`, `n241_k32_s3-nokl` | 10.000 (continuam de 5.000) | seeds em 241 |
| `n241_k32_s1-nokl` | 20.000 (continua de 10.000) | achatamento é falta de passos? |
| `n128_k32_s2-nokl` | 20.000 (continua de 10.000) | referência em 128 com o mesmo número de passos |
| `n8/n16/n32/n64_k32_s2/s3-nokl` | 5.000 e 10.000 | seeds dos pontos intermediários (vários já terminaram) |

## Previsões registradas antes dos resultados (25/11/2026)

- **Seeds a 10.000 passos:** os novos pontos caem dentro de ±0,15 bit das médias atuais (8: 1,07; 32: 2,11; 128: 2,84; 241: 2,90).
- **20.000 passos:** em N = 128 a seed 2 dá 2,90–3,05 (a seed 1 deu 2,96).
- **Critério para o achatamento:**
  - se for **falta de passos**, N = 241 a 20.000 passos fica pelo menos 0,2 bit acima de N = 128 a 20.000 passos (≥ 3,16 se 128 ficar em 2,96);
  - se for **outra coisa** (frases por locutor ou limite real), a diferença fica abaixo de 0,1 bit.
  - Entre 0,1 e 0,2 bit, a conclusão espera mais seeds.

## Quando terminar

Escrever o resultado (próximo número livre em `resultados/`), atualizar R16 com uma nota se a média mudar, e mover este arquivo para `../concluido/`.
