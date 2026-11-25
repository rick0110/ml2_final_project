# Caminho 03 — Receita de treino

**Estado:** ativo

## Pergunta

Quanto dos resultados depende da receita de treino (KL, duração, otimização), e não de N ou da arquitetura? Uma comparação justa entre N exige treinos perto da saturação.

## Hipótese atual

A rampa do KL comprimia o latente e jogava fora a voz ([R15](../../resultados/R15_rampa_do_kl.md)). Sem ela, o treino satura perto de 10.000–20.000 passos em N = 128 ([R18](../../resultados/R18_duracao_do_treino.md)), e mais locutores pedem mais passos ([R16](../../resultados/R16_curva_de_N_sem_kl.md)).

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| 23/11 | Treinar 5.000 passos piorava os bits (1,75 → 1,42) | [R15](../../resultados/R15_rampa_do_kl.md) |
| 24/11 | A causa é a rampa do KL; sem ela, 1,81 → 2,44 → 2,84 bits (2.500 → 5.000 → 10.000) | [R15](../../resultados/R15_rampa_do_kl.md) |
| 25/11 | 20.000 passos em N = 128: 2,96 (+0,12 sobre 10.000) | [R18](../../resultados/R18_duracao_do_treino.md) |

## Em andamento

- [`em_andamento/duracao_do_treino.md`](em_andamento/duracao_do_treino.md): curva de duração em N = 128 e 241.

## Ideias ainda não testadas

- Decaimento da taxa de aprendizado no fim do treino (hoje é constante).
- KL pequeno e constante (ou *free bits*), em vez de zero, para manter o latente regular sem apagar a voz.
- Batch maior: a GPU usa 9 de 20 GB.
