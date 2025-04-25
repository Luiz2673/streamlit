import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

st.set_page_config( #configura a página para que tenha o page title como classi... e layout wide para ocupara a tela toda 
    page_title= "Classificação de Veículos",
    #layout="wide"
)

@st.cache_data # Evita que se repita a mesma função (no caso a de baixo) toda vez que houver alteração 
def load_data_and_model():
    carros = pd.read_csv("car.csv", sep=",") # lendo a planilha car.csv com separadores de , nas colunas
    encoder = OrdinalEncoder() # variavel que define como ordinalencoder 

    for col in carros.columns.drop('class'): # criasse a váriavel col que percorre carros removendo a coluna class (que é a variavel independente)
        carros[col] = carros[col].astype('category') # carros na posição de col se torna categóricos 

# aqui ele transforma o X e Y em valores categóricos, até ontde entendi será criado um identificador numerico para cada tipo 
    X_encoded = encoder.fit_transform(carros.drop('class',axis=1)) #axis 1 = coluna 

    y = carros['class'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42) #X variavel independete e y variavel dependente, test_size define que vai ser usado 30% para teste e os outros 70% para treino e o randon_state faz a divisão dos registros 

    modelo = CategoricalNB() #aqui é a IA propriamente dita
    modelo.fit(X_train,y_train) # aqui ele esta usando as variaveis que irão treinar a IA para ela prever a classe (lembrando que no for excluimos a coluna classe, assim teremos como ver se a previsão dele esta correta sem que ele consulte a planilha)

    y_pred = modelo.predict(X_test) # aqui ele fará a previsão do X_test que esta recebendo o X_encoded
    acuracia = accuracy_score(y_test,y_pred) # aqui ele vai trazer a acuracia de acerto em porcentagem

    return encoder, modelo, acuracia, carros #retorno das variáveis 

encoder, modelo, acuracia, carros = load_data_and_model()

st.title("Previsão de Qualidade de Veículo") #titulo da pagina (h1)
st.write(f"Acurácia do modelo: {acuracia:.2f}")# Impressão da acurácia recebendo 2 casas decimais depois do valor 

input_features = [ #lista para pegar os valor de cada st 
    st.selectbox("Preço:", carros['buying'].unique()), #input se lista com o nome preço que recebe de carros a coluna buying onde se repete uma unica vez cada opção existente na planilha 
    st.selectbox("Manutenção:", carros['maint'].unique()),
    st.selectbox("Portas:", carros['doors'].unique()),
    st.selectbox("Capacidade:", carros['persons'].unique()),
    st.selectbox("Porta Malas:", carros['lug_boot'].unique()),
    st.selectbox("Segurança:", carros['safety'].unique())
]

if st.button("Processar"): #criação do botão 
    input_df = pd.DataFrame([input_features], columns=carros.columns.drop('class')) #chama o metodo de data frame com os dados da lista acima, com os nomes da coluna correto, ele pega os mesmos nomes de coluna de carros excluindo a coluna class 
    input_encoded = encoder.transform(input_df) # transforma dados categóricos em numeros usando o encoder que foi feito na função 
    predict_encoded = modelo.predict(input_encoded) # passa os dados de entrada codificados onde ele vai ter o dado de previsão dos veiculos, ele vai retornar um numero
    previsao = carros['class'].astype('category').cat.categories[predict_encoded][0]
    st.header(f"Resultado da previsão: {previsao}")
