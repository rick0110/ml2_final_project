# Caminho 05 — Representação da voz (codificador, latente, referência)

**Estado:** quase encerrado

## Pergunta

A cópia de vozes novas é limitada pela forma como a voz é descrita: o que o latente carrega, de onde vem o vetor, quantas referências se usam, quantas dimensões?

## Conclusão até aqui

Não. Mudar a representação não reduz a necessidade de locutores: a voz chega ao decodificador (4–5 bits no latente, [R10](../../resultados/R10_bits_gargalo_do_decodificador.md)), e o limite é ele conseguir renderizá-la. A exceção é encolher demais o latente (2–8 dimensões), que vira um gargalo próprio.

## Histórico

| Quando | O que aprendemos | Resultado |
|---|---|---|
| Colab | Troca de referência e contrastiva: nada muda | [R02](../../resultados/R02_troca_de_referencia_e_contrastiva.md) |
| Colab | Vetor de locutor pré-treinado: ignorado | [R03](../../resultados/R03_vetor_de_locutor_pre_treinado.md) |
| Colab | Dropout de estilo: vetor usado, sem ganho | [R04](../../resultados/R04_dropout_de_estilo.md) |
| Colab | Média de várias referências: não ajuda nas 38 vozes; baixa o WER | [R07](../../resultados/R07_varias_referencias.md) |
| 23–24/11 | Latente de 8: −0,40 bit (descontando a reinicialização); latente de 2 zera os bits | [R13](../../resultados/R13_latente_menor_e_controle.md) |

## Em andamento

- [`em_andamento/controle_de_reinicializacao.md`](em_andamento/controle_de_reinicializacao.md): o custo de reinicializar as cabeças do VAE em outros N.
