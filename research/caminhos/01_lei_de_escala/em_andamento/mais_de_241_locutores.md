# Mais de 241 locutores (planejado, não iniciado)

**Caminho:** [01](../README.md)

## Objetivo

Testar se a lei continua acima de 241 locutores, o máximo do LibriTTS train-clean-100. É o teste mais forte da lei: extrapolação para N que nenhum ajuste viu.

## O que falta

1. **Corpus maior.** O LibriTTS train-clean-360 tem cerca de 900 locutores. Hoje só há, numa das máquinas, a versão **LibriTTS-R** (áudio restaurado) de train-clean-100, train-clean-360 e test-clean, baixada para o [caminho 08](../../08_flow_matching/README.md). Misturar LibriTTS-R com o LibriTTS original muda o domínio do áudio. Para manter a comparação, há duas opções:
   - baixar o train-clean-360 original;
   - refazer a curva inteira em LibriTTS-R (mais caro).
2. **Pré-processamento** igual ao do train-clean-100 (`scripts/preprocess/`) e manifestos com `coverage_manifest.py`, com seleção aninhada (os 241 atuais dentro dos 512).
3. **Runs:** `n512_k32` e `n900_k32` sem KL, 10.000 e 20.000 passos, 2–3 seeds.

## Previsões registradas (25/11/2026, antes de qualquer treino)

Pela inclinação sem KL a 10.000 passos (0,38–0,41 bit por duplicação, [R16](../../../resultados/R16_curva_de_N_sem_kl.md)), partindo de 128 locutores (2,84):

- N = 512: 3,6 bits se a lei continuar; 2,9–3,1 se o achatamento de 241 for real.
- N = 900: 3,9 bits se a lei continuar.
- **A lei cai** se N = 512 ficar abaixo de 3,1 bits com o mesmo número de passos e o mesmo K.
