# Caminho 02 — Gargalo do decodificador

**Estado:** ativo

## Pergunta

Onde a informação da voz se perde, e o que muda se o decodificador for condicionado de outra forma?

## Hipótese atual

O decodificador é o gargalo: referência e latente carregam 4–5 bits, a saída bem menos ([R10](../../resultados/R10_bits_gargalo_do_decodificador.md)). Com poucos locutores o limite é cobertura (o decodificador ouviu poucas vozes). Com muitos, o modo de injetar o estilo também limita, e condicionar cada passo do decodificador (FiLM) ajuda. Ainda não está claro se o FiLM eleva o teto ou só acelera o aprendizado.

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| Colab | Codificador e latente não são o problema; o decodificador perde ~4 bits | [R10](../../resultados/R10_bits_gargalo_do_decodificador.md) |
| 23–24/11 | FiLM: ganho nulo em N = 8, +0,31 em 128, +0,44 em 241 (2.500 passos) | [R12](../../resultados/R12_film.md) |
| 24–25/11 | FiLM sem KL: 3,08 bits em 241 a 10.000 passos (melhor resultado); a 10.000 passos a vantagem sobre "sem KL" diminui | [R17](../../resultados/R17_film_sem_kl.md) |

## Em andamento

- [`em_andamento/film_sem_kl_seeds.md`](em_andamento/film_sem_kl_seeds.md): seeds e 20.000 passos para separar "teto mais alto" de "aprende mais rápido".

## Ideias ainda não testadas

- Condicionar também a atenção e a pós-rede.
- Arquitetura em que o estilo entra em todas as camadas por construção (ver [caminho 08](../08_flow_matching/README.md)).
