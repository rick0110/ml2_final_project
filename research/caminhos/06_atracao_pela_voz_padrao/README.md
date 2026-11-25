# Caminho 06 — Atração pela voz padrão

**Estado:** pausado

## Pergunta

As 2.000 frases da voz padrão (LJSpeech) puxam as saídas para ela? Quanto da "voz não copiada" é atração real e quanto é artefato da medida, que compara com a voz padrão?

## Conclusão até aqui

A medida binária (`sim_ref > sim_default`) distorce: a margem é m = (½ − ρ)·‖e_ref − e_0‖², então vozes parecidas com a voz padrão trocam de lado por ruído ([R09](../../resultados/R09_vozes_dificeis.md)). Cortar a voz padrão pela metade sobe a cópia 4 pontos, mas quase não muda os bits ([R14](../../resultados/R14_metade_da_voz_padrao.md)). Por isso a métrica principal passou a ser os bits de voz, que não usam a voz padrão.

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| Colab | Vozes agudas e perto da voz padrão falham mais; parte é artefato da medida | [R09](../../resultados/R09_vozes_dificeis.md) |
| 23–24/11 | Metade da voz padrão: +4 pontos de cópia, bits iguais (8 seeds) | [R14](../../resultados/R14_metade_da_voz_padrao.md) |

## Em andamento

Nada. Ver [`em_andamento/README.md`](em_andamento/README.md).
