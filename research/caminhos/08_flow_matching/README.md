# Caminho 08 — Arquitetura moderna: flow matching (F5-TTS)

**Estado:** pausado

## Pergunta

A lei de escala em N e o gargalo do decodificador valem numa arquitetura moderna? Aqui, flow matching com um transformer de difusão (F5-TTS), que gera o espectrograma inteiro de uma vez, condiciona todas as camadas e escala melhor que o Tacotron 2.

## Por que importa

O Tacotron 2 é lento e não escala. Se a lei medida aqui se repetir numa arquitetura atual, ela vira um resultado sobre clonagem de voz em geral, e não sobre um modelo antigo. Isso também orienta quantos locutores e quanta computação uma nova arquitetura precisaria.

## Plano

1. Modelos pequenos em 2–3 tamanhos (por exemplo 4–8 camadas, dimensão 256–512), como os laboratórios fazem para prever escala com pouco recurso.
2. Subconjuntos com o mesmo total de horas e N variando, com seleção aninhada como no Tacotron.
3. A mesma avaliação de bits de voz nas mesmas 38 vozes externas ([protocolo](../../protocolo.md)).

## Histórico

Sem resultados ainda.

## Em andamento

- [`em_andamento/preparacao_f5.md`](em_andamento/preparacao_f5.md)
