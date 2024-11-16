import streamlit as st
import io

import numpy as np
import pandas as pd

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import matplotlib.pyplot as plt
import seaborn as sns

import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


st.set_page_config(
    page_title="Projeto #02 | Previsão de Renda",
    page_icon="https://github.com/digslima/ebac-image/blob/main/newebac_logo_black_half.png?raw=true",
    layout="wide",
    initial_sidebar_state="auto",
)


st.sidebar.markdown('''
<div style="text-align:center">
<img src="https://github.com/digslima/ebac-image/blob/main/newebac_logo_black_half.png?raw=true" width=50%>
</div>

# **Profissão: Cientista de Dados**
### [**Projeto #02** | Previsão de renda](https://github.com/digslima/ebac_projeto_semantix.git)


---
''', unsafe_allow_html=True)

with st.sidebar.expander("Índice", expanded=False):
    st.markdown('''
    - [Etapa 1 CRISP - DM: Entendimento do negócio](#1)
    - [Etapa 2 Crisp-DM: Entendimento dos dados](#2)
        > - [Dicionário de dados](#dicionario)
        > - [Carregando os pacotes](#pacotes)
        > - [Carregando os dados](#dados)
        > - [Entendimento dos dados - Univariada](#univariada)
        >> - [Estatísticas descritivas das variáveis quantitativas](#describe)
        > - [Entendimento dos dados - Bivariadas](#bivariada)
        >> - [Matriz de correlação](#correlacao)
        >> - [Matriz de dispersão](#dispersao)
        >>> - [Clustermap](#clustermap)
        >>> - [Linha de tendência](#tendencia)
        >> - [Análise das variáveis qualitativas](#qualitativas)
    - [Etapa 3 Crisp-DM: Preparação dos dados](#3)
    - [Etapa 4 Crisp-DM: Modelagem](#4)
        > - [Divisão da base em treino e teste](#train_test)
        > - [Seleção de hiperparâmetros do modelo com for loop](#for_loop)
        > - [Rodando o modelo](#rodando)
    - [Etapa 5 Crisp-DM: Avaliação dos resultados](#5)
    - [Etapa 6 Crisp-DM: Implantação](#6)
        > - [Simulação](#simulacao)
    ''', unsafe_allow_html=True)


with st.sidebar.expander("Bibliotecas/Pacotes", expanded=False):
    st.code('''
    import streamlit as st
    import io

    import numpy as np
    import pandas as pd

    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report

    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import tree
    import streamlit as st

    ''', language='python')


st.markdown('# <div style="text-align:center"> [Previsão de renda](https://github.com/digslima/ebac_projeto_semantix.git) </div>',
            unsafe_allow_html=True)


st.divider()


st.markdown('''
## Etapa 1 CRISP - DM: Entendimento do negócio <a name="1"></a>
''', unsafe_allow_html=True)


st.markdown('''
Uma instituição financeira deseja compreender melhor o perfil de renda de seus novos clientes para diversos propósitos, como definir de forma mais precisa o limite de crédito dos cartões, sem precisar solicitar holerites ou documentos que possam afetar a experiência do cliente.

Para alcançar esse objetivo, a instituição realizou um estudo com alguns clientes, verificando suas rendas por meio de holerites e outros documentos. Com base nisso, pretende desenvolver um modelo preditivo para estimar a renda dos clientes utilizando algumas variáveis já disponíveis em seu banco de dados.''')


st.markdown('''
## Etapa 2 Crisp-DM: Entendimento dos dados<a name="2"></a>
''', unsafe_allow_html=True)


st.markdown('''
### Dicionário de dados <a name="dicionario"></a>

| Variável              | Descrição                                                                                                  | Tipo             |
| --------------------- |:----------------------------------------------------------------------------------------------------------:| ----------------:|
| data_ref              | Data de referência de coleta das variáveis                                                                 | object           |
| id_cliente            | Código identificador exclusivo do cliente                                                                  | int              |
| sexo                  | Sexo do cliente (M = 'Masculino'; F = 'Feminino')                                                          | object (binária) |
| posse_de_veiculo      | Indica se o cliente possui veículo (True = 'Possui veículo'; False = 'Não possui veículo')                 | bool (binária)   |
| posse_de_imovel       | Indica se o cliente possui imóvel (True = 'Possui imóvel'; False = 'Não possui imóvel')                    | bool (binária)   |
| qtd_filhos            | Quantidade de filhos do cliente                                                                            | int              |
| tipo_renda            | Tipo de renda do cliente (Empresário, Assalariado, Servidor público, Pensionista, Bolsista)                | object           |
| educacao              | Grau de instrução do cliente (Primário, Secundário, Superior incompleto, Superior completo, Pós graduação) | object           |
| estado_civil          | Estado civil do cliente (Solteiro, União, Casado, Separado, Viúvo)                                         | object           |
| tipo_residencia       | Tipo de residência do cliente (Casa, Governamental, Com os pais, Aluguel, Estúdio, Comunitário)            | object           |
| idade                 | Idade do cliente em anos                                                                                   | int              |
| tempo_emprego         | Tempo no emprego atual                                                                                     | float            |
| qt_pessoas_residencia | Quantidade de pessoas que moram na residência                                                              | float            |
| **renda**             | Valor numérico decimal representando a renda do cliente em reais                                           | float            |
''', unsafe_allow_html=True)


st.markdown('''
### Carregando os dados <a name="dados"></a>
''', unsafe_allow_html=True)


# VERIFICAR ARQUIVOS LOCAIS:
# path_to_find = os.listdir()
# st.title(path_to_find)
filepath = 'previsao_de_renda.csv'
renda = pd.read_csv(filepath_or_buffer=filepath)

buffer = io.StringIO()
renda.info(buf=buffer)
st.text(buffer.getvalue())
st.dataframe(renda)


st.table(renda.nunique()
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'Variável',
                               0: 'Valores únicos'}))


renda.drop(columns=['Unnamed: 0', 'id_cliente'], inplace=True)
st.write('Quantidade total de linhas:',
         len(renda))
st.write('Quantidade de linhas duplicadas:',
         renda.duplicated().sum())
st.write('Quantidade após remoção das linhas duplicadas:',
         len(renda.drop_duplicates()))
renda.drop_duplicates(inplace=True, ignore_index=True)
buffer = io.StringIO()
renda.info(buf=buffer)
st.text(buffer.getvalue())


st.markdown('''
### Entendimento dos dados - Univariada <a name="univariada"></a>
''', unsafe_allow_html=True)


with st.expander("PAnálise Exploratória de Dados", expanded=True):
    os.makedirs('./output', exist_ok=True)

renda.describe().to_csv('./output/renda_descriptive_statistics.csv')

for column in renda.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(renda[column], bins=30, kde=True)
    plt.title(f"Distribuição de {column}")
    plt.savefig(f'./output/histograma_{column}.png')
    plt.close()

for column in renda.select_dtypes(include='object').columns:
    plt.figure(figsize=(8, 6))
    renda[column].value_counts().plot(kind='bar')
    plt.title(f"Contagem de {column}")
    plt.savefig(f'./output/barra_{column}.png')
    plt.close()

num_cols = renda.select_dtypes(include='number').columns
sns.pairplot(renda[num_cols])
plt.savefig('./output/pairplot.png')
plt.close()

numeric_data = renda.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.savefig('./output/heatmap_correlacao.png')
plt.close()


st.markdown('''
####  Estatísticas descritivas das variáveis quantitativas <a name="describe"></a>
''', unsafe_allow_html=True)


st.write(renda.describe().transpose())


st.markdown('''
### Entendimento dos dados - Bivariadas <a name="bivariada"></a>
''', unsafe_allow_html=True)


st.markdown('''
#### Matriz de correlação <a name="correlacao"></a>
''', unsafe_allow_html=True)


numeric_data = renda.iloc[:, 3:].select_dtypes(include='number')

correlation_tail = numeric_data.corr().iloc[-1, :]

st.write("Última linha da matriz de correlação:", correlation_tail)


st.markdown('A partir da matriz de correlação, é possível observar que a variável que apresenta maior relação com a varíavel renda é tempo_emprego, com um índice de correlação de 38,5%.')


st.markdown('''
#### Matriz de dispersão <a name="dispersao"></a>
''', unsafe_allow_html=True)


sns.pairplot(data=renda,
             hue='tipo_renda',
             vars=['qtd_filhos',
                   'idade',
                   'tempo_emprego',
                   'qt_pessoas_residencia',
                   'renda'],
             diag_kind='hist')
st.pyplot(plt)


st.markdown('Ao examinar o **pairplot**, que é uma matriz de dispersão, é possível identificar alguns *outliers* na variável de renda. Esses *outliers*, embora ocorram com pouca frequência, podem influenciar os resultados da análise de tendência. Além disso, observa-se uma baixa correlação entre quase todas as variáveis quantitativas, o que confirma os resultados obtidos na matriz de correlação.')


st.markdown('''
##### Clustermap <a name="clustermap"></a>
''', unsafe_allow_html=True)

numeric_renda = renda.select_dtypes(include=[float, int])
corr_matrix = numeric_renda.corr()

cmap = sns.diverging_palette(h_neg=100,
                             h_pos=359,
                             as_cmap=True,
                             sep=1,
                             center='light')
ax = sns.clustermap(data=corr_matrix, 
                    figsize=(10, 10), 
                    center=0, 
                    cmap=cmap)
plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45)
st.pyplot(plt)


st.markdown('Usando o *clustermap*, podemos confirmar que a maioria das variáveis apresenta baixa correlação com a renda. A única exceção notável é a variável `tempo_emprego`, que demonstra um índice de correlação significativo. Além disso, as variáveis booleanas`posse_de_imove` e `posse_de_veicul` também foram analisadas, mas exibem uma correlação reduzida com a renda.')


st.markdown('''
#####  Linha de tendência <a name="tendencia"></a>
''', unsafe_allow_html=True)


plt.figure(figsize=(16, 9))
sns.scatterplot(x='tempo_emprego',
                y='renda',
                hue='tipo_renda',
                size='idade',
                data=renda,
                alpha=0.4)
sns.regplot(x='tempo_emprego',
            y='renda',
            data=renda,
            scatter=False,
            color='.3')
st.pyplot(plt)

plt.figure(figsize=(16,9))


st.markdown('Apesar de a correlação entre as variáveis `tempo_empreg` e `renda` não ser muito elevada, a inclinação da linha de tendência permite identificar claramente uma covariância positiva entre elas.')


st.markdown('''
#### Análise das variáveis qualitativas <a name="qualitativas"></a>
''', unsafe_allow_html=True)


with st.expander("Análise de relevância preditiva com variáveis booleanas", expanded=True):
    plt.rc('figure', figsize=(12, 4))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.pointplot(x='posse_de_imovel',
                  y='renda',
                  data=renda,
                  dodge=True,
                  ax=axes[0])
    sns.pointplot(x='posse_de_veiculo',
                  y='renda',
                  data=renda,
                  dodge=True,
                  ax=axes[1])
    st.pyplot(plt)

    plt.rc('figure', figsize=(12,4))
fig, axes = plt.subplots(nrows=1, ncols=2)


st.markdown('Ao analisar os gráficos acima, percebe-se que a variável posse_de_veículo tem maior relevância na predição de renda. Isso é evidenciado pela maior distância entre os intervalos de confiança para quem possui ou não um veículo. Em contraste, a variável posse_de_imóvel não mostra diferença significativa entre as diferentes condições de posse imobiliária.')


with st.expander("Análise das variáveis qualitativas ao longo do tempo", expanded=True):
    renda['data_ref'] = pd.to_datetime(arg=renda['data_ref'])

    qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns

    plt.rc('figure', figsize=(16, 4))

    for col in qualitativas:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.subplots_adjust(wspace=.6)
    
        tick_labels = renda['data_ref'].map(lambda x: x.strftime('%b/%Y')).unique()
        tick_indices = range(len(tick_labels))
    
        renda_crosstab = pd.crosstab(index=renda['data_ref'], 
                                 columns=renda[col], 
                                 normalize='index')
        ax0 = renda_crosstab.plot.bar(stacked=True, ax=axes[0])
        ax0.set_xticks(tick_indices)
        ax0.set_xticklabels(labels=tick_labels, rotation=45)
        axes[0].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
    
        ax1 = sns.pointplot(x='data_ref', y='renda', hue=col, data=renda, dodge=True, errorbar=('ci', 95), ax=axes[1])
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(labels=tick_labels, rotation=45)
        axes[1].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        st.pyplot(plt)
    

st.markdown('''
## Etapa 3 Crisp-DM: Preparação dos dados<a name="3"></a>
''', unsafe_allow_html=True)


renda.drop(columns='data_ref', inplace=True)
renda.dropna(inplace=True)
st.table(pd.DataFrame(index=renda.nunique().index,
                      data={'tipos_dados': renda.dtypes,
                            'qtd_valores': renda.notna().sum(),
                            'qtd_categorias': renda.nunique().values}))


with st.expander("Conversão das variáveis categóricas em variáveis numéricas (dummies)", expanded=True):
    renda_dummies = pd.get_dummies(data=renda)
    buffer = io.StringIO()
    renda_dummies.info(buf=buffer)
    st.text(buffer.getvalue())

    st.table((renda_dummies.corr()['renda']
              .sort_values(ascending=False)
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'var',
                               'renda': 'corr'})
              .style.bar(color=['darkred', 'darkgreen'], align=0)
              ))


st.markdown('''
## Etapa 4 Crisp-DM: Modelagem <a name="4"></a>
''', unsafe_allow_html=True)


st.markdown('A técnica selecionada foi o **DecisionTreeRegressor**, devido à sua habilidade em lidar com problemas de regressão, como a previsão de renda dos clientes. Além disso, árvores de decisão são intuitivas, fáceis de interpretar e auxiliam na identificação dos atributos mais importantes para a previsão da variável-alvo, tornando-a uma opção adequada para este projeto.')


st.markdown('''
### Divisão da base em treino e teste <a name="train_test"></a>
''', unsafe_allow_html=True)


X = renda_dummies.drop(columns='renda')
y = renda_dummies['renda']
st.write('Quantidade de linhas e colunas de X:', X.shape)
st.write('Quantidade de linhas de y:', len(y))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
st.write('X_train:', X_train.shape)
st.write('X_test:', X_test.shape)
st.write('y_train:', y_train.shape)
st.write('y_test:', y_test.shape)


st.markdown('''
### Seleção de hiperparâmetros do modelo com for loop <a name="for_loop"></a>
''', unsafe_allow_html=True)


score = pd.DataFrame({'max_depth': pd.Series(dtype='int'), 
                      'min_samples_leaf': pd.Series(dtype='int'), 
                      'score': pd.Series(dtype='float')})

for x in range(1, 21):
    for y in range(1, 31):
        reg_tree = DecisionTreeRegressor(random_state=42, 
                                         max_depth=x, 
                                         min_samples_leaf=y)
        reg_tree.fit(X_train, y_train)
        
        score = pd.concat(objs=[score, 
                                pd.DataFrame({'max_depth': [x], 
                                              'min_samples_leaf': [y], 
                                              'score': [reg_tree.score(X=X_test, y=y_test)]})], 
                          axis=0, 
                          ignore_index=True)

score.sort_values(by='score', ascending=False, inplace=True)


st.markdown('''
### Rodando o modelo <a name="rodando"></a>
''', unsafe_allow_html=True)


reg_tree = DecisionTreeRegressor(random_state=42,
                                 max_depth=8,
                                 min_samples_leaf=4)
# reg_tree.fit(X_train, y_train)
st.text(reg_tree.fit(X_train, y_train))



with st.expander("Visualização gráfica da árvore com plot_tree", expanded=True):
    plt.figure(figsize=(18, 9))
    tree.plot_tree(decision_tree=reg_tree,
                   feature_names=X.columns,
                   filled=True)
    st.pyplot(plt)


with st.expander("Visualização impressa da árvore", expanded=False):
    text_tree_print = tree.export_text(decision_tree=reg_tree)
    st.text(text_tree_print)


st.markdown('''
## Etapa 5 Crisp-DM: Avaliação dos resultados <a name="5"></a>
''', unsafe_allow_html=True)


r2_train = reg_tree.score(X=X_train, y=y_train)
r2_test = reg_tree.score(X=X_test, y=y_test)
template = 'O coeficiente de determinação (𝑅2) da árvore com profundidade = {0} para a base de {1} é: {2:.2f}'
st.write(template.format(reg_tree.get_depth(),
                         'treino',
                         r2_train)
         .replace(".", ","))
st.write(template.format(reg_tree.get_depth(),
                         'teste',
                         r2_test)
         .replace(".", ","))


renda['renda_predict'] = np.round(reg_tree.predict(X), 2)
st.dataframe(renda[['renda', 'renda_predict']])


st.markdown('''
## Etapa 6 Crisp-DM: Implantação <a name="6"></a>
''', unsafe_allow_html=True)


'---'
