# Custo de reinicializar as cabeças do VAE em outros N

**Início:** 24/11/2026 · **Caminho:** [05](../README.md)

## Objetivo

Em [R13](../../../resultados/R13_latente_menor_e_controle.md), reinicializar as cabeças do VAE custou 0,78 bit em N = 128. Medir esse custo em N = 32 e 241 permite comparar o latente de 8 dimensões com o controle certo em toda a curva.

## Runs (na fila ou rodando)

`n32_k32_s1/s2-zr32` e `n241_k32_s1/s2-zr32`, a 2.500 passos (receita original).

## Previsões registradas antes dos resultados (25/11/2026)

- O custo da reinicialização (original − zr32) fica entre 0,4 e 0,9 bit em N = 32 e em 241.
- zr32 fica acima de z8 em todos os N (z8: 0,62 em N = 32, 0,42 em 241).

## Depois

Se o custo for parecido em todos os N, este caminho se encerra com uma nota em R13. A pergunta do tamanho do latente só volta se um treino longo sem KL ([caminho 03](../../03_receita_de_treino/README.md)) recuperar a reinicialização.
