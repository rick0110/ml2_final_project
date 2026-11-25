# Caminho 07 — Aumento de vozes

**Estado:** bloqueado

## Pergunta

Se o que conta é o número de vozes reais, dá para fabricar vozes para o treino sem gravar locutores novos?

## Conclusão até aqui

Mudar a velocidade das gravações não funciona: as pseudo-vozes não contam como locutores, como a teoria previu ([R05](../../resultados/R05_pseudo_locutores.md)). A versão que deveria funcionar é converter as frases da voz padrão para centenas de vozes reais com um conversor de voz treinado sem transcrição.

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| Colab | Pseudo-locutores por velocidade: N efetivo não muda | [R05](../../resultados/R05_pseudo_locutores.md) |

## Por que está bloqueado

Os scripts estão prontos (`scripts/coverage/vc_cascade.py`, `vc_voices.py`, variante `-vcaug`), com previsões registradas na seção 7 de [`reports/teoria_cobertura.md`](../../../reports/teoria_cobertura.md). Eles usam um conversor de voz pré-treinado de terceiros (kNN-VC, via `torch.hub`), e o controle de permissões do ambiente barrou rodá-lo. Depende de autorização do autor do projeto.

## Em andamento

Nada. Ver [`em_andamento/README.md`](em_andamento/README.md).
