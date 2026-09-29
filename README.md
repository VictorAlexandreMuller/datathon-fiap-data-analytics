# Datathon Passos Mágicos — Pós-Tech

## Sobre o projeto

Este projeto foi desenvolvido para o Datathon da Pós-Tech e utiliza
dados da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE)
da Associação Passos Mágicos.

O objetivo é analisar a trajetória educacional dos estudantes entre
2022 e 2024 e desenvolver uma solução preditiva capaz de apoiar a
identificação antecipada de risco de defasagem.

---

## Objetivos

O trabalho contempla:

- análise da adequação ao nível (IAN);
- desempenho acadêmico (IDA);
- engajamento (IEG);
- autoavaliação (IAA);
- aspectos psicossociais (IPS);
- aspectos psicopedagógicos (IPP);
- Ponto de Virada (IPV);
- análise multidimensional do INDE;
- evolução longitudinal dos estudantes;
- identificação de padrões de defasagem;
- desenvolvimento de modelo preditivo;
- disponibilização do modelo em Streamlit.

---

## Dados

Foram utilizados dados PEDE referentes a:

- 2022
- 2023
- 2024

---

## Estrutura analítica

### Notebook 01 — Preparação dos dados

Responsável por:

- auditoria da base;
- padronização das variáveis;
- validação do IAN;
- construção da estrutura longitudinal;
- criação do target futuro.

### Notebook 02 — Análise e storytelling

Responsável por responder às questões analíticas do Datathon e
produzir os principais insights.

### Notebook 03 — Machine Learning

Responsável por:

- feature engineering;
- divisão temporal de treino e teste;
- comparação entre modelos;
- avaliação preditiva;
- definição do threshold operacional;
- exportação do modelo final.

---

## Estratégia de Machine Learning

Foi utilizada validação temporal:

- 2022 → 2023: treinamento;
- 2023 → 2024: teste.

Foram comparados:

- Regressão Logística;
- Random Forest.

O Random Forest apresentou melhor desempenho e foi selecionado como
modelo final.

---

## Resultados do modelo

Threshold operacional: 0,35

| Métrica | Resultado |
|---|---:|
| Accuracy | 76,7% |
| Precision | 67,8% |
| Recall | 80,5% |
| F1-score | 73,6% |
| ROC-AUC | 87,1% |
| PR-AUC | 82,4% |

O modelo identificou aproximadamente 8 em cada 10 estudantes que
posteriormente apresentaram defasagem no conjunto temporal de teste.

---

## Aplicação

A aplicação Streamlit permite informar os indicadores atuais de um
estudante e obter a probabilidade estimada de risco de defasagem no
ano seguinte.


---

## Principais tecnologias

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## Observações

O modelo é uma ferramenta de apoio à análise e não substitui avaliação
pedagógica, psicopedagógica ou profissional.

Os resultados são associações e previsões baseadas nos dados
históricos disponíveis e não devem ser interpretados como evidências
de causalidade.

---

## Links

Aplicação Streamlit - [Clique aqui](https://datathon-fiap-data-analytics-id6r8skv4pmvkhbrzdpxgr.streamlit.app/)

Aplicação YouTube - [Clique aqui](https://youtu.be/uZrXtE5SOeI)
