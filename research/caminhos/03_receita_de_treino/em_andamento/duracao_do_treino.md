# Duração do treino sem KL

**Início:** 25/11/2026 · **Caminho:** [03](../README.md)

## Objetivo

Medir onde o treino sem KL satura, para fixar a duração dos próximos experimentos. Os runs são os de 20.000 passos do [caminho 01](../../01_lei_de_escala/em_andamento/curva_longa_sem_kl.md) e do [caminho 02](../../02_gargalo_do_decodificador/em_andamento/film_sem_kl_seeds.md): `n128_k32_s2-nokl`, `n241_k32_s1-nokl` e `n241_k32_s1-filmnokl`.

## Previsões registradas antes dos resultados (25/11/2026)

- **N = 128:** de 10.000 para 20.000 passos, ganho de 0,05 a 0,20 bit (a seed 1 ganhou 0,12).
- **N = 241:** o ganho de 10.000 para 20.000 passos é maior que em 128, porque mais locutores pedem mais treino ([R16](../../../resultados/R16_curva_de_N_sem_kl.md)).
- **Decisão:** se em N = 241 o ganho passar de 0,2 bit, os próximos experimentos com N ≥ 128 usam 20.000 passos; senão, 10.000.
