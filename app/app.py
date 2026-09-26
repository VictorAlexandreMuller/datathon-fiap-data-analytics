from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# CONFIGURAÇÃO DA PÁGINA

st.set_page_config(
    page_title="Passos Mágicos | Risco de Defasagem",
    page_icon="⭐",
    layout="wide"
)


# CAMINHOS

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

CAMINHO_MODELO = (
    ROOT /
    "models" /
    "modelo_risco.joblib"
)

CAMINHO_METADATA = (
    ROOT /
    "models" /
    "metadata_modelo.json"
)

CAMINHO_METRICAS = (
    ROOT /
    "outputs" /
    "tables" /
    "metricas_modelo_final.csv"
)

CAMINHO_MATRIZ = (
    ROOT /
    "outputs" /
    "figures" /
    "modelo_matriz_confusao.png"
)


# CARREGAMENTO

@st.cache_resource
def carregar_modelo():

    modelo = joblib.load(
        CAMINHO_MODELO
    )

    with open(
        CAMINHO_METADATA,
        "r",
        encoding="utf-8"
    ) as arquivo:

        metadata = json.load(
            arquivo
        )

    return modelo, metadata


try:

    modelo, metadata = carregar_modelo()

except Exception as erro:

    st.error(
        "Não foi possível carregar o modelo preditivo."
    )

    st.exception(erro)

    st.stop()


# INFORMAÇÕES DO MODELO

FEATURES = metadata.get(
    "features",
    [
        "fase_num",
        "idade",
        "ian",
        "ida",
        "ieg",
        "iaa",
        "ips",
        "ipv",
        "inde",
        "defasagem"
    ]
)

THRESHOLD = float(
    metadata.get(
        "threshold",
        0.35
    )
)


# CABEÇALHO

st.title(
    "⭐ Sistema Preditivo de Risco de Defasagem"
)

st.write(
    """
    Aplicação desenvolvida para apoiar a identificação preventiva
    de estudantes com maior probabilidade de apresentar defasagem
    no ano seguinte.
    """
)

st.info(
    """
    O resultado deve ser utilizado como instrumento de apoio à
    análise da equipe. Ele não substitui avaliação pedagógica,
    psicopedagógica ou profissional.
    """
)


# ABAS

aba_predicao, aba_modelo, aba_metricas = st.tabs(
    [
        "Predição",
        "Sobre o modelo",
        "Métricas"
    ]
)


# ABA 1 — PREDIÇÃO

with aba_predicao:

    st.subheader(
        "Avaliação de risco"
    )

    st.write(
        """
        Informe os indicadores disponíveis para o estudante.
        O sistema calculará a probabilidade estimada de
        defasagem no ano seguinte.
        """
    )


    # BLOCO 1

    col1, col2, col3 = st.columns(3)

    with col1:

        fase_num = st.selectbox(
            "Fase atual",
            options=list(range(0, 10)),
            index=1,
            help=(
                "Fase atual do estudante na "
                "Passos Mágicos."
            )
        )

    with col2:

        idade = st.number_input(
            "Idade",
            min_value=6,
            max_value=30,
            value=12,
            step=1
        )

    with col3:

        defasagem = st.number_input(
            "Defasagem atual",
            min_value=-10.0,
            max_value=5.0,
            value=-1.0,
            step=1.0,
            help=(
                "Fase efetiva menos fase ideal. "
                "Valores negativos indicam defasagem."
            )
        )


    # BLOCO 2

    st.markdown(
        "#### Indicadores educacionais"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        ian = st.selectbox(
            "IAN — Adequação ao nível",
            options=[
                2.5,
                5.0,
                10.0
            ],
            index=1
        )

    with col2:

        ida = st.number_input(
            "IDA — Desempenho acadêmico",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

    with col3:

        ieg = st.number_input(
            "IEG — Engajamento",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )


    # BLOCO 3

    col1, col2, col3 = st.columns(3)

    with col1:

        iaa = st.number_input(
            "IAA — Autoavaliação",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    with col2:

        ips = st.number_input(
            "IPS — Psicossocial",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    with col3:

        ipv = st.number_input(
            "IPV — Ponto de Virada",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )


    # BLOCO 4

    inde = st.number_input(
        "INDE — Índice de Desenvolvimento Educacional",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


    # BOTÃO DE PREDIÇÃO

    st.markdown("---")

    calcular = st.button(
        "Calcular probabilidade de risco",
        type="primary",
        use_container_width=True
    )


    if calcular:

        valores = {
            "fase_num": fase_num,
            "idade": idade,
            "ian": ian,
            "ida": ida,
            "ieg": ieg,
            "iaa": iaa,
            "ips": ips,
            "ipv": ipv,
            "inde": inde,
            "defasagem": defasagem
        }


        # Garante a mesma ordem usada no treinamento
        entrada = pd.DataFrame(
            [
                {
                    feature:
                        valores[feature]

                    for feature in FEATURES
                }
            ]
        )


        probabilidade = (
            modelo.predict_proba(
                entrada
            )[0, 1]
        )


        alerta = (
            probabilidade >= THRESHOLD
        )


        st.markdown("---")

        st.subheader(
            "Resultado"
        )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Probabilidade estimada",
                f"{probabilidade:.1%}"
            )

        with col2:

            st.metric(
                "Threshold operacional",
                f"{THRESHOLD:.0%}"
            )

        with col3:

            if alerta:

                st.metric(
                    "Classificação",
                    "ALERTA"
                )

            else:

                st.metric(
                    "Classificação",
                    "SEM ALERTA"
                )


        st.progress(
            min(
                float(probabilidade),
                1.0
            )
        )


        if alerta:

            st.warning(
                """
                **Alerta preventivo identificado.**

                A probabilidade estimada ultrapassou o
                threshold operacional do modelo.

                Recomenda-se que a equipe utilize esse
                sinal para priorizar uma análise mais
                detalhada da situação do estudante.
                """
            )

        else:

            st.success(
                """
                **O modelo não identificou alerta preventivo
                neste momento.**

                Isso não significa ausência de risco.
                O acompanhamento educacional deve continuar
                normalmente.
                """
            )

        # DADOS UTILIZADOS

        with st.expander(
            "Ver dados utilizados na predição"
        ):

            tabela_entrada = (
                entrada
                .T
                .reset_index()
            )

            tabela_entrada.columns = [
                "Variável",
                "Valor"
            ]

            st.dataframe(
                tabela_entrada,
                use_container_width=True,
                hide_index=True
            )


# ABA 2 — SOBRE O MODELO

with aba_modelo:

    st.subheader(
        "Como o modelo foi desenvolvido"
    )

    st.markdown(
        """
        O modelo foi construído utilizando uma estratégia
        de **validação temporal**.

        **Treinamento**

        Dados dos estudantes em **2022** foram utilizados
        para prever a situação observada em **2023**.

        **Teste**

        Dados de **2023** foram utilizados para prever
        a situação observada em **2024**.

        Essa abordagem reduz o risco de utilizar
        informações futuras durante o treinamento e
        aproxima a avaliação da utilização real do sistema.
        """
    )


    st.markdown(
        "### Modelo selecionado"
    )

    st.write(
        """
        Foram comparados dois algoritmos:

        - Regressão Logística
        - Random Forest

        O **Random Forest** apresentou o melhor desempenho
        no conjunto temporal de teste e foi selecionado
        como modelo final.
        """
    )


    st.markdown(
        "### Variável-alvo"
    )

    st.latex(
        r"""
        Risco_{t+1} =
        \begin{cases}
        1, & Defasagem_{t+1} < 0 \\
        0, & Defasagem_{t+1} \geq 0
        \end{cases}
        """
    )


    st.markdown(
        "### Threshold operacional"
    )

    st.write(
        f"""
        O modelo utiliza **{THRESHOLD:.0%}** como
        threshold para geração do alerta.

        Esse valor foi escolhido após comparar diferentes
        thresholds e buscar equilíbrio entre **precision,
        recall e F1-score**, com prioridade para a
        identificação preventiva dos estudantes em risco.
        """
    )


    st.markdown(
        "### Variáveis utilizadas"
    )

    descricoes = {
        "fase_num":
            "Fase atual do estudante",

        "idade":
            "Idade",

        "ian":
            "Indicador de Adequação ao Nível",

        "ida":
            "Indicador de Desempenho Acadêmico",

        "ieg":
            "Indicador de Engajamento",

        "iaa":
            "Indicador de Autoavaliação",

        "ips":
            "Indicador Psicossocial",

        "ipv":
            "Indicador de Ponto de Virada",

        "inde":
            "Índice de Desenvolvimento Educacional",

        "defasagem":
            "Diferença entre fase efetiva e fase ideal"
    }

    tabela_features = pd.DataFrame(
        {
            "Variável": FEATURES,
            "Descrição": [
                descricoes.get(
                    feature,
                    feature
                )

                for feature in FEATURES
            ]
        }
    )

    st.dataframe(
        tabela_features,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### Limitações"
    )

    st.markdown(
        """
        - O modelo foi treinado em dados históricos da
          Passos Mágicos e sua validade está condicionada
          a populações semelhantes.
        - Existem valores ausentes em algumas variáveis,
          especialmente no conjunto de teste.
        - Há relação estrutural entre IAN e defasagem.
        - As importâncias do Random Forest não representam
          causalidade.
        - O modelo deve apoiar, e não substituir, a análise
          realizada pelos profissionais da instituição.
        """
    )


# ABA 3 — MÉTRICAS

with aba_metricas:

    st.subheader(
        "Desempenho do modelo"
    )


    # TENTA LER AS MÉTRICAS EXPORTADAS

    if CAMINHO_METRICAS.exists():

        metricas = pd.read_csv(
            CAMINHO_METRICAS
        ).iloc[0]

        accuracy = metricas["accuracy"]
        precision = metricas["precision"]
        recall = metricas["recall"]
        f1 = metricas["f1"]
        roc_auc = metricas["roc_auc"]
        pr_auc = metricas["pr_auc"]

    else:

        # Valores obtidos na avaliação final
        accuracy = 0.767
        precision = 0.678
        recall = 0.805
        f1 = 0.736
        roc_auc = 0.871
        pr_auc = 0.824


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.1%}"
        )

        st.metric(
            "Precision",
            f"{precision:.1%}"
        )

    with col2:

        st.metric(
            "Recall",
            f"{recall:.1%}"
        )

        st.metric(
            "F1-score",
            f"{f1:.1%}"
        )

    with col3:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.1%}"
        )

        st.metric(
            "PR-AUC",
            f"{pr_auc:.1%}"
        )


    st.markdown("---")

    st.markdown(
        """
        **Resultado operacional principal**

        Com threshold de **35%**, o modelo identificou
        aproximadamente **80,5% dos estudantes que
        efetivamente apresentaram defasagem no período
        seguinte**.
        """
    )


    # MATRIZ DE CONFUSÃO

    st.markdown(
        "### Matriz de confusão"
    )

    if CAMINHO_MATRIZ.exists():

        st.image(
            str(CAMINHO_MATRIZ),
            use_container_width=False
        )

    else:

        matriz = pd.DataFrame(
            [
                [339, 118],
                [60, 248]
            ],
            columns=[
                "Predito sem risco",
                "Predito com risco"
            ],
            index=[
                "Real sem risco",
                "Real com risco"
            ]
        )

        st.dataframe(
            matriz,
            use_container_width=True
        )


    st.markdown(
        """
        - **339** estudantes sem risco foram corretamente
          classificados.
        - **248** estudantes em risco foram corretamente
          identificados.
        - **118** alertas foram falsos positivos.
        - **60** estudantes em risco não foram identificados.
        """
    )


# RODAPÉ

st.markdown("---")

st.caption(
    """
    Datathon Passos Mágicos — Pós-Tech |
    Modelo desenvolvido para fins acadêmicos e de apoio
    à análise educacional.
    """
)