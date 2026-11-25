# Caminho 04 — Locutores vs frases por locutor

**Estado:** ativo

## Pergunta

Com a mesma quantidade de dados, vale mais ter muitas vozes com poucas frases ou poucas vozes com muitas frases? E as frases por locutor (K) explicam o achatamento em 241 locutores?

## Hipótese atual

Para vozes novas, o número de vozes pesa mais que as frases por voz ([R01](../../resultados/R01_grade_N_x_K.md), [R19](../../resultados/R19_locutores_vs_frases.md)). Mas K também ajuda com N fixo, e com N grande K = 32 pode estar limitando.

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| Colab | K decide as vozes treinadas; N decide as novas; 32 × 8 copia tanto quanto 8 × 128 | [R01](../../resultados/R01_grade_N_x_K.md) |
| 24–25/11 | Em bits, com 1.024 frases: 8 locutores 0,74; 32 locutores 1,23; 128 locutores 1,36 (original); o mesmo sem KL | [R19](../../resultados/R19_locutores_vs_frases.md) |

## Em andamento

- [`em_andamento/k64_sem_kl.md`](em_andamento/k64_sem_kl.md): K = 64 sem KL e a grade de K na receita original.
