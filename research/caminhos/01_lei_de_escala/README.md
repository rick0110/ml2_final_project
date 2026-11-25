# Caminho 01 — Lei de escala em N

**Estado:** ativo

## Pergunta

Quantos bits da voz de referência chegam ao áudio gerado em função do número de locutores de treino (N)? A relação é regular (uma lei)? Satura, e onde?

## Hipótese atual

Os bits crescem linearmente em log2 N (cada duplicação de locutores acrescenta uma quantidade fixa de bits), porque o decodificador passa a alcançar mais vozes diferentes (cobertura), com o mesmo ruído por voz ([R10](../../resultados/R10_bits_gargalo_do_decodificador.md)). A inclinação depende da receita de treino; o aparente teto em 241 locutores pode ser falta de treino ou de frases por locutor, e não um limite real.

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| Colab | Locutores decidem as vozes novas; frases decidem as treinadas | [R01](../../resultados/R01_grade_N_x_K.md) |
| Colab | Com 4 vozes de teste, não dava para ver ganho acima de 128 locutores (ruído de seed) | [R06](../../resultados/R06_128_e_200_locutores.md) |
| Colab | Em 38 vozes, o ganho de 32 para 128 é real | [R08](../../resultados/R08_avaliacao_larga_38_vozes.md) |
| Colab | Em bits: 0,31 bit por duplicação, com dois pontos | [R10](../../resultados/R10_bits_gargalo_do_decodificador.md) |
| 23/11 | Com 3 seeds: 0,27 ± 0,03 bit por duplicação até 128; sem ganho de 128 para 241 | [R11](../../resultados/R11_curva_de_N_receita_original.md) |
| 24–25/11 | Sem a rampa do KL e com 10.000 passos: 0,41 ± 0,06 por duplicação; 2,90 bits em 241, ainda achatando | [R16](../../resultados/R16_curva_de_N_sem_kl.md) |

## Em andamento

- [`em_andamento/curva_longa_sem_kl.md`](em_andamento/curva_longa_sem_kl.md): seeds a 10.000 passos e 20.000 passos em 241 locutores.
- [`em_andamento/mais_de_241_locutores.md`](em_andamento/mais_de_241_locutores.md): planejado; exige preparar outro corpus.

## Perguntas em aberto

- O achatamento de 128 para 241 é falta de passos, de frases por locutor ou limite real?
- A lei continua acima de 241 locutores? Na taxa atual, quantos locutores levariam a saída ao nível da fala real (4,2–4,7 bits)?
