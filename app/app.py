from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# CONFIGURAÇÃO DA PÁGINA

st.set_page_config(
    page_title="Passos Mágicos | Datathon",
    page_icon="⭐",
    layout="wide"
)


# ESTILO

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                rgba(25, 76, 130, 0.10),
                rgba(255, 153, 0, 0.10)
            );
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin-bottom: 0.4rem;
        }

        .story-box {
            padding: 1.3rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin: 1rem 0;
        }

        .small-muted {
            opacity: 0.75;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# CAMINHOS

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

PASTA_MODELOS = ROOT / "models"
PASTA_TABELAS = ROOT / "outputs" / "tables"
PASTA_FIGURAS = ROOT / "outputs" / "figures"

CAMINHO_MODELO = (
    PASTA_MODELOS /
    "modelo_risco.joblib"
)

CAMINHO_METADATA = (
    PASTA_MODELOS /
    "metadata_modelo.json"
)

CAMINHO_METRICAS = (
    PASTA_TABELAS /
    "metricas_modelo_final.csv"
)

CAMINHO_MATRIZ = (
    PASTA_FIGURAS /
    "modelo_matriz_confusao.png"
)


# ARQUIVOS DAS ANÁLISES

TABELAS = {
    "q1": PASTA_TABELAS / "q1_defasagem.csv",
    "q2": PASTA_TABELAS / "q2_ida.csv",
    "q3": PASTA_TABELAS / "q3_ieg.csv",
    "q4": PASTA_TABELAS / "q4_iaa.csv",
    "q5": PASTA_TABELAS / "q5_ips.csv",
    "q6": PASTA_TABELAS / "q6_ipp.csv",
    "q7": PASTA_TABELAS / "q7_ipv.csv",
    "q8": PASTA_TABELAS / "q8_inde.csv",
    "q10": PASTA_TABELAS / "q10_evolucao.csv",
    "q11": PASTA_TABELAS / "q11_defasagem_por_fase.csv",
}

FIGURAS = {
    "q1": PASTA_FIGURAS / "01_defasagem_por_ano.png",
    "q2": PASTA_FIGURAS / "02_ida_por_fase.png",
    "q11": PASTA_FIGURAS / "11_defasagem_por_fase.png",
    "matriz": PASTA_FIGURAS / "modelo_matriz_confusao.png",
}


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


@st.cache_data
def carregar_csv(caminho):

    caminho = Path(caminho)

    if caminho.exists():

        return pd.read_csv(
            caminho
        )

    return None


def mostrar_tabela(caminho):

    tabela = carregar_csv(
        str(caminho)
    )

    if tabela is None:

        st.warning(
            f"Arquivo não encontrado: {caminho.name}"
        )

        return None

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True
    )

    return tabela


def mostrar_imagem(caminho, legenda=None):

    if caminho.exists():

        st.image(
            str(caminho),
            caption=legenda
        )

    else:

        st.info(
            f"Gráfico não encontrado: {caminho.name}"
        )


# MODELO

try:

    modelo, metadata = carregar_modelo()

except Exception as erro:

    st.error(
        "Não foi possível carregar o modelo preditivo."
    )

    st.exception(erro)

    st.stop()


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


# MÉTRICAS DO MODELO

metricas_df = carregar_csv(
    str(CAMINHO_METRICAS)
)

if metricas_df is not None:

    metricas = metricas_df.iloc[0]

    ACCURACY = float(
        metricas["accuracy"]
    )

    PRECISION = float(
        metricas["precision"]
    )

    RECALL = float(
        metricas["recall"]
    )

    F1 = float(
        metricas["f1"]
    )

    ROC_AUC = float(
        metricas["roc_auc"]
    )

    PR_AUC = float(
        metricas["pr_auc"]
    )

else:

    # Fallback com os resultados obtidos no Notebook 03

    ACCURACY = 0.767
    PRECISION = 0.678
    RECALL = 0.805
    F1 = 0.736
    ROC_AUC = 0.871
    PR_AUC = 0.824


# CABEÇALHO

st.markdown(
    """
    <div class="hero">
        <h1>⭐ Datathon Passos Mágicos</h1>
        <h3>Dados para compreender o passado e antecipar riscos</h3>
        <p>
            Análise longitudinal dos dados PEDE de 2022 a 2024
            e desenvolvimento de um sistema preditivo de apoio
            à identificação de risco de defasagem.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ABAS

(
    aba_storytelling,
    aba_analises,
    aba_predicao,
    aba_modelo
) = st.tabs(
    [
        "Storytelling",
        "Análises",
        "Predição",
        "Modelo"
    ]
)


# ABA 1 — STORYTELLING

with aba_storytelling:

    st.header(
        "Da análise histórica ao alerta preventivo"
    )

    st.markdown(
        """
        O ponto de partida deste projeto foi uma pergunta:

        ### É possível identificar o risco de defasagem antes que ele se consolide?

        Para responder, analisamos a evolução dos estudantes
        da Associação Passos Mágicos entre 2022 e 2024,
        investigamos os principais indicadores educacionais
        e transformamos os padrões encontrados em uma
        solução preditiva.
        """
    )


    # RESUMO DOS DADOS

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Registros analisados",
            "3.030"
        )

    with col2:

        st.metric(
            "Período",
            "2022–2024"
        )

    with col3:

        st.metric(
            "Presentes nos 3 anos",
            "468 alunos"
        )

    with col4:

        st.metric(
            "Recall do modelo",
            f"{RECALL:.1%}"
        )


    # CAPÍTULO 1

    st.markdown("---")

    st.subheader(
        "1. A adequação ao nível melhorou"
    )

    q1 = carregar_csv(
        str(TABELAS["q1"])
    )

    if q1 is not None:

        q1["ano"] = pd.to_numeric(
            q1["ano"],
            errors="coerce"
        )

        def obter_q1(ano, faixa):

            linha = q1[
                (q1["ano"] == ano) &
                (
                    q1["faixa_defasagem"]
                    == faixa
                )
            ]

            if len(linha) == 0:
                return None

            return float(
                linha.iloc[0]["percentual"]
            )


        fase_2022 = obter_q1(
            2022,
            "Em fase / adiantado"
        )

        fase_2024 = obter_q1(
            2024,
            "Em fase / adiantado"
        )

        severa_2022 = obter_q1(
            2022,
            "Defasagem severa"
        )

        severa_2024 = obter_q1(
            2024,
            "Defasagem severa"
        )


        col1, col2 = st.columns(2)

        with col1:

            if fase_2022 is not None:

                st.metric(
                    "Em fase / adiantados — 2022",
                    f"{fase_2022:.1f}%"
                )

        with col2:

            if fase_2024 is not None:

                st.metric(
                    "Em fase / adiantados — 2024",
                    f"{fase_2024:.1f}%"
                )


        if (
            severa_2022 is not None
            and severa_2024 is not None
        ):

            st.markdown(
                f"""
                Entre 2022 e 2024, a proporção de alunos em
                fase ou adiantados avançou de
                **{fase_2022:.1f}% para {fase_2024:.1f}%**.

                No mesmo período, a defasagem severa caiu de
                **{severa_2022:.1f}% para
                {severa_2024:.1f}%**.
                """
            )


    mostrar_imagem(
        FIGURAS["q1"],
        "Evolução do perfil de defasagem entre 2022 e 2024"
    )


    # CAPÍTULO 2

    st.markdown("---")

    st.subheader(
        "2. Mas a melhora não ocorreu da mesma forma em todas as fases"
    )

    q11 = carregar_csv(
        str(TABELAS["q11"])
    )

    if q11 is not None:

        q11["ano"] = pd.to_numeric(
            q11["ano"],
            errors="coerce"
        )

        q11["fase_num"] = pd.to_numeric(
            q11["fase_num"],
            errors="coerce"
        )

        dados_2024 = q11[
            q11["ano"] == 2024
        ]

        fase0 = dados_2024[
            dados_2024["fase_num"] == 0
        ]

        fase1 = dados_2024[
            dados_2024["fase_num"] == 1
        ]

        col1, col2 = st.columns(2)

        if len(fase0):

            with col1:

                st.metric(
                    "Defasados na Fase 0 — 2024",
                    (
                        f"{float(fase0.iloc[0]['percentual_defasados']):.1f}%"
                    )
                )

        if len(fase1):

            with col2:

                st.metric(
                    "Defasados na Fase 1 — 2024",
                    (
                        f"{float(fase1.iloc[0]['percentual_defasados']):.1f}%"
                    )
                )


    st.markdown(
        """
        A melhora global não significa que o desafio esteja
        distribuído uniformemente.

        As fases iniciais ainda concentram uma parcela
        importante dos estudantes defasados, indicando
        pontos prioritários para acompanhamento.
        """
    )


    mostrar_imagem(
        FIGURAS["q11"],
        "Percentual de alunos defasados por fase"
    )


    # CAPÍTULO 3

    st.markdown("---")

    st.subheader(
        "3. O engajamento aparece ligado à trajetória educacional"
    )

    q3 = carregar_csv(
        str(TABELAS["q3"])
    )

    if q3 is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "IEG × IDA",
                (
                    f"{q3['IEG_x_IDA'].min():.2f}"
                    " a "
                    f"{q3['IEG_x_IDA'].max():.2f}"
                )
            )

        with col2:

            st.metric(
                "IEG × IPV",
                (
                    f"{q3['IEG_x_IPV'].min():.2f}"
                    " a "
                    f"{q3['IEG_x_IPV'].max():.2f}"
                )
            )


    st.markdown(
        """
        Em todos os anos analisados, o engajamento
        apresentou associação positiva moderada tanto com
        o desempenho acadêmico quanto com o Ponto de Virada.

        Isso torna o IEG um sinal relevante para compreender
        a trajetória do aluno, ainda que associação não
        signifique causalidade.
        """
    )


    # CAPÍTULO 4

    st.markdown("---")

    st.subheader(
        "4. O Ponto de Virada reúne diferentes dimensões"
    )

    q7 = carregar_csv(
        str(TABELAS["q7"])
    )

    if q7 is not None:

        q7["ano"] = pd.to_numeric(
            q7["ano"],
            errors="coerce"
        )

        q7_2024 = (
            q7[
                q7["ano"] == 2024
            ]
            .sort_values(
                "correlacao_com_IPV",
                ascending=False
            )
        )

        if len(q7_2024):

            st.dataframe(
                q7_2024[
                    [
                        "indicador",
                        "correlacao_com_IPV"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    st.markdown(
        """
        IDA, IEG e IPP aparecem entre os indicadores mais
        associados ao IPV.

        Em 2024, o IPP apresentou uma associação
        particularmente elevada com o Ponto de Virada,
        reforçando o caráter multidimensional da trajetória
        educacional.
        """
    )


    # CAPÍTULO 5

    st.markdown("---")

    st.subheader(
        "5. A pergunta então muda"
    )

    st.markdown(
        """
        Até aqui, os dados permitem responder:

        > **Quem está defasado e quais padrões estão associados a essa trajetória?**

        A partir disso, surge uma pergunta mais útil para intervenção:

        > ### Quem apresenta sinais de que poderá estar defasado no próximo período?
        """
    )


    # CAPÍTULO 6

    st.markdown("---")

    st.subheader(
        "6. Transformando análise em previsão"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "ROC-AUC",
            f"{ROC_AUC:.1%}"
        )

    with col2:

        st.metric(
            "Recall",
            f"{RECALL:.1%}"
        )

    with col3:

        st.metric(
            "F1-score",
            f"{F1:.1%}"
        )


    st.success(
        """
        Com o threshold operacional de 35%, o modelo
        identificou **248 dos 308 estudantes que
        posteriormente apresentaram defasagem**.

        Isso corresponde a aproximadamente **8 em cada 10
        casos futuros de defasagem** no conjunto temporal
        de teste.
        """
    )


    # CAPÍTULO 7

    st.markdown("---")

    st.subheader(
        "7. Da previsão para a ação"
    )

    st.markdown(
        """
        **Indicadores atuais**

        ↓

        **Modelo preditivo**

        ↓

        **Probabilidade estimada de risco**

        ↓

        **Alerta preventivo**

        ↓

        **Priorização da análise pela equipe**
        """
    )


    st.info(
        """
        O objetivo não é automatizar decisões sobre o aluno.

        O modelo funciona como uma ferramenta de triagem,
        ajudando a equipe a identificar mais cedo quais
        estudantes podem merecer acompanhamento adicional.
        """
    )


    st.markdown(
        """
        ### Recomendações

        **1. Priorizar o acompanhamento das fases iniciais**

        Elas continuam concentrando níveis elevados de
        defasagem.

        **2. Acompanhar engajamento, desempenho e Ponto de
        Virada**

        Esses indicadores aparecem de forma consistente na
        trajetória educacional observada.

        **3. Utilizar o modelo como alerta preventivo**

        A probabilidade de risco deve apoiar a priorização,
        e não substituir a avaliação dos profissionais.
        """
    )


    st.markdown(
        """
        ---
        ### Da medição do passado à antecipação de onde a atenção pode ser necessária.
        """
    )


# ABA 2 — ANÁLISES

with aba_analises:

    st.header(
        "Análises do Datathon"
    )

    st.write(
        """
        Esta área reúne as análises utilizadas para
        responder às principais questões propostas no case.
        """
    )


    analise = st.selectbox(
        "Selecione uma análise",
        [
            "1 — Adequação ao nível (IAN)",
            "2 — Desempenho acadêmico (IDA)",
            "3 — Engajamento (IEG)",
            "4 — Autoavaliação (IAA)",
            "5 — Aspectos psicossociais (IPS)",
            "6 — Aspectos psicopedagógicos (IPP)",
            "7 — Ponto de Virada (IPV)",
            "8 — Multidimensionalidade do INDE",
            "10 — Efetividade do programa",
            "11 — Insight adicional: defasagem por fase"
        ]
    )


    # Q1

    if analise.startswith("1 —"):

        st.subheader(
            "Adequação ao nível — IAN"
        )

        st.markdown(
            """
            **Pergunta**

            Qual é o perfil geral de defasagem dos alunos e
            como ele evolui ao longo dos anos?
            """
        )

        mostrar_imagem(
            FIGURAS["q1"]
        )

        mostrar_tabela(
            TABELAS["q1"]
        )

        st.success(
            """
            A proporção de alunos em fase ou adiantados
            aumentou de aproximadamente **30,1% em 2022
            para 53,8% em 2024**.

            A defasagem severa caiu de aproximadamente
            **3,3% para 0,3%** no mesmo período.
            """
        )


    # Q2

    elif analise.startswith("2 —"):

        st.subheader(
            "Desempenho acadêmico — IDA"
        )

        st.markdown(
            """
            **Pergunta**

            O desempenho acadêmico médio está melhorando,
            estagnado ou caindo ao longo das fases e anos?
            """
        )

        mostrar_imagem(
            FIGURAS["q2"]
        )

        mostrar_tabela(
            TABELAS["q2"]
        )

        st.info(
            """
            O IDA médio avançou de **6,09 em 2022 para
            6,66 em 2023**, seguido de uma retração para
            **6,35 em 2024**.

            Mesmo com o recuo, 2024 permaneceu acima do
            nível médio observado em 2022.

            A evolução não foi uniforme entre as fases.
            """
        )


    # Q3

    elif analise.startswith("3 —"):

        st.subheader(
            "Engajamento — IEG"
        )

        st.markdown(
            """
            **Pergunta**

            O engajamento apresenta relação com desempenho
            acadêmico e Ponto de Virada?
            """
        )

        mostrar_tabela(
            TABELAS["q3"]
        )

        st.success(
            """
            O IEG apresentou associação positiva moderada e
            consistente com IDA e IPV nos três anos.

            A relação com o IPV foi ligeiramente superior
            à relação com o desempenho acadêmico.
            """
        )


    # Q4

    elif analise.startswith("4 —"):

        st.subheader(
            "Autoavaliação — IAA"
        )

        st.markdown(
            """
            **Pergunta**

            A percepção do aluno sobre si mesmo é coerente
            com desempenho acadêmico e engajamento?
            """
        )

        mostrar_tabela(
            TABELAS["q4"]
        )

        st.info(
            """
            As correlações entre IAA e os indicadores
            objetivos foram fracas.

            Em 2022 e 2024, aproximadamente **85% dos
            estudantes apresentaram IAA superior ao IDA**.

            Isso indica que a autoavaliação captura uma
            dimensão diferente do desempenho acadêmico
            isolado.
            """
        )


    # Q5

    elif analise.startswith("5 —"):

        st.subheader(
            "Aspectos psicossociais — IPS"
        )

        st.markdown(
            """
            **Pergunta**

            Há padrões psicossociais que antecedem quedas
            futuras de desempenho ou engajamento?
            """
        )

        mostrar_tabela(
            TABELAS["q5"]
        )

        st.info(
            """
            O IPS isoladamente não apresentou associação
            longitudinal relevante com a variação futura
            de IDA ou IEG.

            As correlações observadas ficaram próximas de
            zero nos dois períodos analisados.
            """
        )


    # Q6

    elif analise.startswith("6 —"):

        st.subheader(
            "Aspectos psicopedagógicos — IPP"
        )

        st.markdown(
            """
            **Pergunta**

            As avaliações psicopedagógicas confirmam a
            defasagem identificada pelo IAN?
            """
        )

        mostrar_tabela(
            TABELAS["q6"]
        )

        st.info(
            """
            O IPP apresentou relação positiva, porém fraca,
            com IAN e defasagem.

            Portanto, os indicadores apresentam algum
            alinhamento, mas não medem exatamente a mesma
            dimensão.
            """
        )


    # Q7

    elif analise.startswith("7 —"):

        st.subheader(
            "Ponto de Virada — IPV"
        )

        st.markdown(
            """
            **Pergunta**

            Quais indicadores apresentam maior associação
            com o Ponto de Virada?
            """
        )

        mostrar_tabela(
            TABELAS["q7"]
        )

        st.success(
            """
            IDA, IEG e IPP foram os indicadores com maior
            associação ao IPV.

            Em 2024, a correlação entre **IPP e IPV chegou
            a 0,705**.
            """
        )


    # Q8

    elif analise.startswith("8 —"):

        st.subheader(
            "Multidimensionalidade do INDE"
        )

        st.markdown(
            """
            **Pergunta**

            Como IDA, IEG, IPS e IPP se relacionam com a
            nota global do aluno?
            """
        )

        mostrar_tabela(
            TABELAS["q8"]
        )

        st.success(
            """
            IDA e IEG apresentaram os maiores coeficientes
            padronizados entre as dimensões analisadas.

            O conjunto formado por IDA + IEG + IPS + IPP
            explicou aproximadamente **82,3% da variação
            observada do INDE**.

            Como o INDE é construído a partir desses
            indicadores, o resultado deve ser interpretado
            como decomposição associativa, e não como
            causalidade.
            """
        )


    # Q10

    elif analise.startswith("10 —"):

        st.subheader(
            "Efetividade e evolução longitudinal"
        )

        st.markdown(
            """
            **Pergunta**

            Os indicadores apresentam melhora consistente
            ao longo do ciclo?
            """
        )

        mostrar_tabela(
            TABELAS["q10"]
        )

        st.info(
            """
            O IAN e a defasagem apresentaram melhora nos
            dois períodos consecutivos.

            IDA e IEG melhoraram entre 2022 e 2023, mas
            recuaram entre 2023 e 2024.

            Portanto, existe evidência de evolução na
            adequação ao nível, mas não uma melhora uniforme
            em todas as dimensões.

            Como os dados são observacionais, não é possível
            atribuir causalidade exclusivamente ao programa.
            """
        )


    # Q11

    elif analise.startswith("11 —"):

        st.subheader(
            "Insight adicional — Defasagem por fase"
        )

        mostrar_imagem(
            FIGURAS["q11"]
        )

        mostrar_tabela(
            TABELAS["q11"]
        )

        st.warning(
            """
            Em 2024, as fases iniciais ainda concentravam
            níveis elevados de defasagem.

            **Fase 0: aproximadamente 75%**

            **Fase 1: aproximadamente 70,8%**

            Esse resultado sugere priorização de
            acompanhamento nessas etapas, sem interpretar
            a fase como causa da defasagem.
            """
        )


# ABA 3 — PREDIÇÃO

with aba_predicao:

    st.header(
        "Predição de risco de defasagem"
    )

    st.write(
        """
        Informe os indicadores atuais do estudante.

        O modelo estima a probabilidade de o aluno
        apresentar defasagem no período seguinte.
        """
    )

    st.info(
        """
        O resultado é um alerta de apoio à decisão.

        Ele não substitui a avaliação pedagógica,
        psicopedagógica ou profissional.
        """
    )


    with st.form(
        "form_predicao"
    ):

        st.subheader(
            "Dados do estudante"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            fase_num = st.selectbox(
                "Fase atual",
                options=list(range(0, 10)),
                index=1
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


        # ----------------------------------------------------
        # IAN CALCULADO AUTOMATICAMENTE
        # ----------------------------------------------------

        if defasagem >= 0:

            ian = 10.0

        elif defasagem >= -2:

            ian = 5.0

        else:

            ian = 2.5


        st.caption(
            f"IAN derivado automaticamente da defasagem: "
            f"{ian}"
        )


        st.subheader(
            "Indicadores educacionais"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            ida = st.number_input(
                "IDA — Desempenho acadêmico",
                min_value=0.0,
                max_value=10.0,
                value=6.0,
                step=0.1
            )

        with col2:

            ieg = st.number_input(
                "IEG — Engajamento",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )

        with col3:

            iaa = st.number_input(
                "IAA — Autoavaliação",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )


        col1, col2, col3 = st.columns(3)

        with col1:

            ips = st.number_input(
                "IPS — Psicossocial",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )

        with col2:

            ipv = st.number_input(
                "IPV — Ponto de Virada",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )

        with col3:

            inde = st.number_input(
                "INDE — Desenvolvimento Educacional",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )


        calcular = st.form_submit_button(
            "Calcular probabilidade de risco",
            type="primary",
            use_container_width=True
        )


    # RESULTADO

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


        faltantes = [
            feature
            for feature in FEATURES
            if feature not in valores
        ]

        if faltantes:

            st.error(
                "Existem features ausentes para o modelo: "
                + ", ".join(faltantes)
            )

        else:

            entrada = pd.DataFrame(
                [
                    {
                        feature:
                            valores[feature]

                        for feature in FEATURES
                    }
                ]
            )


            probabilidade = float(
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

                st.metric(
                    "Classificação",
                    (
                        "ALERTA"
                        if alerta
                        else "SEM ALERTA"
                    )
                )


            st.progress(
                min(
                    probabilidade,
                    1.0
                )
            )


            if alerta:

                st.warning(
                    """
                    ### Alerta preventivo identificado

                    A probabilidade estimada ultrapassou o
                    threshold operacional.

                    Recomenda-se priorizar uma análise mais
                    detalhada da situação do estudante.
                    """
                )

            else:

                st.success(
                    """
                    ### Sem alerta preventivo neste momento

                    A probabilidade estimada ficou abaixo do
                    threshold operacional.

                    Isso não significa ausência absoluta de
                    risco. O acompanhamento deve continuar.
                    """
                )


            if fase_num == 9:

                st.warning(
                    """
                    A Fase 9 possui representação limitada
                    nos dados históricos utilizados para
                    desenvolvimento do modelo. Interprete
                    esta previsão com cautela.
                    """
                )


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


# ABA 4 — MODELO

with aba_modelo:

    st.header(
        "Metodologia e desempenho do modelo"
    )


    # VALIDAÇÃO TEMPORAL

    st.subheader(
        "Validação temporal"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            ### Treinamento

            **2022 → 2023**

            600 estudantes

            Indicadores disponíveis em 2022 foram utilizados
            para prever a situação observada em 2023.
            """
        )

    with col2:

        st.info(
            """
            ### Teste

            **2023 → 2024**

            765 estudantes

            Indicadores de 2023 foram utilizados para prever
            a situação observada em 2024.
            """
        )


    st.markdown(
        """
        Essa estratégia simula uma utilização temporal real
        do modelo e evita uma divisão aleatória que poderia
        misturar passado e futuro.
        """
    )


    # TARGET

    st.subheader(
        "Variável-alvo"
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


    # COMPARAÇÃO DOS MODELOS

    st.subheader(
        "Comparação dos modelos"
    )

    comparacao_modelos = pd.DataFrame(
        {
            "Modelo": [
                "Random Forest",
                "Regressão Logística"
            ],

            "Accuracy": [
                0.783,
                0.728
            ],

            "Precision": [
                0.801,
                0.734
            ],

            "Recall": [
                0.614,
                0.510
            ],

            "F1": [
                0.695,
                0.602
            ],

            "ROC-AUC": [
                0.871,
                0.808
            ],

            "PR-AUC": [
                0.824,
                0.766
            ]
        }
    )


    st.dataframe(
        comparacao_modelos.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.success(
        """
        O **Random Forest** apresentou desempenho superior
        ao modelo de Regressão Logística e foi selecionado
        como modelo final.
        """
    )


    # THRESHOLD

    st.subheader(
        "Escolha do threshold"
    )

    thresholds = pd.DataFrame(
        {
            "Threshold": [
                0.30,
                0.35,
                0.40,
                0.45,
                0.50
            ],

            "Precision": [
                0.628,
                0.678,
                0.713,
                0.764,
                0.801
            ],

            "Recall": [
                0.867,
                0.805,
                0.744,
                0.653,
                0.614
            ],

            "F1": [
                0.729,
                0.736,
                0.728,
                0.704,
                0.695
            ]
        }
    )


    st.dataframe(
        thresholds.style.format(
            {
                "Threshold": "{:.2f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        f"""
        Foi selecionado o threshold de **{THRESHOLD:.0%}**.

        Esse valor apresentou o maior F1 entre os thresholds
        testados e manteve recall superior a 80%, adequado à
        proposta de um sistema preventivo.
        """
    )


    # MÉTRICAS FINAIS

    st.subheader(
        "Desempenho final"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{ACCURACY:.1%}"
        )

        st.metric(
            "Precision",
            f"{PRECISION:.1%}"
        )

    with col2:

        st.metric(
            "Recall",
            f"{RECALL:.1%}"
        )

        st.metric(
            "F1-score",
            f"{F1:.1%}"
        )

    with col3:

        st.metric(
            "ROC-AUC",
            f"{ROC_AUC:.1%}"
        )

        st.metric(
            "PR-AUC",
            f"{PR_AUC:.1%}"
        )


    # MATRIZ

    st.subheader(
        "Matriz de confusão"
    )

    if CAMINHO_MATRIZ.exists():

        st.image(
            str(CAMINHO_MATRIZ)
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
        **339** estudantes sem risco foram classificados
        corretamente.

        **248** estudantes em risco foram identificados
        corretamente.

        **118** alertas foram falsos positivos.

        **60** estudantes em risco não foram identificados.
        """
    )


    # FEATURES

    st.subheader(
        "Importância das variáveis"
    )

    importancias = pd.DataFrame(
        {
            "Variável": [
                "Defasagem",
                "IPV",
                "INDE",
                "Fase",
                "Idade",
                "IAN",
                "IDA",
                "IEG",
                "IAA",
                "IPS"
            ],

            "Importância": [
                0.185464,
                0.146370,
                0.134984,
                0.111752,
                0.110899,
                0.106495,
                0.073813,
                0.069422,
                0.036820,
                0.023980
            ]
        }
    )


    st.bar_chart(
        importancias.set_index(
            "Variável"
        )
    )

    st.dataframe(
        importancias.style.format(
            {
                "Importância": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        """
        A importância das variáveis do Random Forest não
        representa causalidade. Variáveis correlacionadas,
        como IAN e defasagem, também podem dividir
        importância entre si.
        """
    )


    # FEATURES UTILIZADAS

    st.subheader(
        "Variáveis utilizadas na predição"
    )

    descricoes = {
        "fase_num":
            "Fase atual",

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
            "Fase efetiva menos fase ideal"
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


    # LIMITAÇÕES

    st.subheader(
        "Limitações"
    )

    st.markdown(
        """
        - O modelo foi treinado em dados históricos da
          Associação Passos Mágicos.

        - A variável idade apresentou aproximadamente
          **39,7% de ausência no conjunto temporal de
          teste**.

        - O pipeline utiliza imputação pela mediana para
          valores ausentes.

        - IAN e defasagem possuem relação estrutural.

        - Importância de feature não representa relação
          causal.

        - Os resultados observacionais do programa não
          permitem concluir causalidade sem um desenho
          comparativo apropriado.

        - O modelo deve apoiar, e não substituir, decisões
          pedagógicas, psicológicas ou psicopedagógicas.
        """
    )


# RODAPÉ

st.markdown("---")

st.caption(
    """
    Datathon Passos Mágicos — Pós-Tech |
    Análise educacional e modelo preditivo de risco de
    defasagem.
    """
)