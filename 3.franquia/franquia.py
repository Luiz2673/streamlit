import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Previsão Inicial de Custo Para Franquia")

dados = pd.read_csv("3.franquia/slr12.csv", sep=";")

X = dados[['FrqAnual']]
y = dados['CusInic']

modelo = LinearRegression().fit(X,y)

col1, col2 = st.columns(2)

with col1: #coluna 1 da tabela 
    st.header("Dados") # Cabeçalho 
    st.table(dados.head(10)) # Vai mostrar apenas as 10 primeiras linhas do CSV 

with col2:
    st.header("Gráfico de Disperção")
    fig, ax = plt.subplots() # Criação do Gráfico fig representa todo o grafico e o ax um subgrafico onde os dados são impressos 
    ax.scatter(X,y, color='blue') # scatter = tipo de grafico que recebe o X e o y na cor azul 
    ax.plot(X, modelo.predict(X), color='red') # linha de regressão 
    st.pyplot(fig)

st.header("Valor Anual da Franquia: ")
novo_valor = st.number_input("Insira Novo Valor", min_value=1.0, max_value=999999.0, value=1500.0, step=0.01) # Um campo de input para valor, onde vai aparecer a frase Insira Novo Valor, o valor minimo é 1.0 o maximo 9999.... o valor padrão 1500.0 e o step é um botão de + e - para subir 0.01 da quantidade na tela
processar = st.button("Procesar") # Botão de processar com o nome que aparece para o usuario 

if processar:
    dados_novo_valor = pd.DataFrame([[novo_valor]], columns=['FrqAnual'])
    prev = modelo.predict(dados_novo_valor)
    st.header(f"Previsão de Custo Inicial R$: {prev[0]:.2f}")