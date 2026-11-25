# R09 — Quais vozes falham: vozes perto da voz padrão, e vozes longe das treinadas quando N é pequeno

- **Data:** novembro de 2026 (fase do Colab)
- **Caminho:** [06 — atração pela voz padrão](../caminhos/06_atracao_pela_voz_padrao/README.md)
- **Veredito:** descritivo; parte da dificuldade é artefato da medida binária

## Pergunta

A falha em copiar uma voz é acaso ou é propriedade da voz? O que caracteriza as vozes difíceis?

## Montagem

Sem treino novo: correlações de postos sobre os 38 locutores externos dos treinos de [R08](R08_avaliacao_larga_38_vozes.md) (`scripts/coverage/voice_hardness.py`).

## Resultado

- **A dificuldade é da voz:** a taxa de cópia por locutor se correlaciona 0,75 entre N = 8 e 32, 0,74 entre 32 e 128 e 0,59 entre 8 e 128.
- **Vozes agudas falham mais:** correlação de −0,53, −0,52 e −0,37 entre a cópia e o F0 da referência (a voz padrão é feminina).
- **A cópia sobe com a distância da referência à voz padrão:** 0,47, 0,58 e 0,53.
- **A distância à voz treinada mais próxima prevê a falha com poucos locutores e deixa de prever com muitos:** −0,33 (N = 8), −0,40 (32), +0,13 (128).

## Conclusão

Com poucos locutores o decodificador só reproduz vozes perto das que ouviu (leitura de cobertura); com 128, quase toda voz tem uma vizinha, e o que sobra de falha é a atração pela voz padrão. A medida binária amplifica isso: com vetores normalizados, a margem é m = (½ − ρ)·‖e_ref − e_0‖², então vozes parecidas com a voz padrão trocam de lado por ruído (seção 8.10 da nota teórica). Por isso a métrica principal passou a ser os bits de voz ([R10](R10_bits_gargalo_do_decodificador.md)).

## Dados

Seções 8.9 e 8.10 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md).
