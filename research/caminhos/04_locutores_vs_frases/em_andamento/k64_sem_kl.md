# K = 64 sem KL, e a grade de K na receita original

**Início:** 25/11/2026 · **Caminho:** [04](../README.md)

## Objetivo

1. Saber se o achatamento de 128 para 241 locutores ([R16](../../../resultados/R16_curva_de_N_sem_kl.md)) é falta de frases por locutor.
2. Completar a grade N × K com seeds, para ajustar uma lei com os dois fatores.

## Runs (na fila ou rodando)

| Run | Passos | Para quê |
|---|---|---|
| `n241_k64_s1/s2-nokl`, `n128_k64_s1/s2-nokl` | 10.000 | K maior em N grande |
| `n64_k64_s1-nokl`, `n32_k64_s1-nokl` | 10.000 | K maior em N pequeno |
| `n128_k16_s1/s2/s3`, `n241_k8_s1/s2` | 2.500 (original) | grade de K |
| `n8_k128_s2/s3`, `n32_k128_s2/s3` | 2.500 (original) | seeds da grade |
| `n8_k128_s2-nokl`, `n32_k128_s2-nokl`, `n128_k8_s2-nokl`, `n241_k8_s1-nokl` | 5.000 | seeds sem KL |

Atenção: o `coverage_manifest.py` avisa quando um locutor não tem K frases e usa as que houver. Com K = 64, parte dos 241 locutores pode ter menos; conferir o aviso no log antes de interpretar.

## Previsões registradas antes dos resultados (25/11/2026)

- **Se o achatamento é falta de frases:** `n241_k64` a 10.000 passos fica pelo menos 0,2 bit acima de `n241_k32` (≥ 3,1) e a diferença entre 128 e 241 aumenta.
- **Se não é:** `n241_k64` fica entre 2,9 e 3,05.
- **Com N fixo:** de K = 32 para 64 o ganho é menor que o de N = 128 para 256 com K fixo. Este último só poderá ser medido no [caminho 01](../../01_lei_de_escala/em_andamento/mais_de_241_locutores.md); até lá, comparar com a inclinação por duplicação de N (0,38–0,41 bit).
