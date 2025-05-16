#Creamos el archivo de la APP en el interprete principal (Phyton)

#####################################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from urllib.parse import parse_qs
import numpy as np
from PIL import Image

st.markdown("""
<style>
.header-banner {
    background-color: #004080;  
    padding: 20px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: white;
}
.header-banner img {
    height: 80px;
    margin-right: 40px;
}
.header-banner h1 {
    flex-grow: 1;
    text-align: center;
    font-size: 28px;
    margin: 0;
}

[data-testid="stApp"] {
    background-color: #e6f0ff; 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}


.button-container {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    justify-content: center;
    margin-bottom: 10px;
}
.cyber-btn {
    background-color: #111;
    color: white;
    border: 1px solid #333;
    padding: 14px 28px;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    width: auto;
    min-width: 150px;  /* <-- ya corregido */
    text-align: center;
    white-space: nowrap;
}
.cyber-btn:hover {
    background-color: #333;
}
</style>

<div class="header-banner">
    <img src="https://img.freepik.com/vector-gratis/ilustracion-bandera-argentina_53876-27120.jpg" alt="Logo">
    <h1>📊 Dashboard de Argentina </h1>
</div>
""", unsafe_allow_html=True)
######################################################
#Definimos la instancia
@st.cache_resource

######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_csv("Argentina_sin_valores_nulos.csv")

   #Selecciono las columnas tipo numericas del dataframe
   numeric_df = df.select_dtypes(['float','int'])  #Devuelve Columnas
   numeric_cols= numeric_df.columns                #Devuelve lista de Columnas

   #Selecciono las columnas tipo texto del dataframe
   text_df = df.select_dtypes(['object'])  #Devuelve Columnas
   text_cols= text_df.columns              #Devuelve lista de Columnas

   #Selecciono algunas columnas categoricas de valores para desplegar en diferentes cuadros
   #categorical_column_sex= df['host_name']

   #Obtengo los valores unicos de la columna categórica seleccionada
   #unique_categories_sex= categorical_column_sex.unique()

   return df, numeric_cols, text_cols, numeric_df

###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, numeric_cols, text_cols,  numeric_df = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
#Widget 1: Selectbox


#TITULO GENERAL
st.title("Buenos Aires")
st.header("selecciona alguna opcion del Menu")



# 1. Inicializamos la vista seleccionada
if "view" not in st.session_state:
    st.session_state.view = "view1"

# 2. Definimos botones como navegación


st.markdown('<div class="button-container">', unsafe_allow_html=True)
cols = st.columns(7)
labels = [
    "Univariado", "Dispersión", "Pastel",
    "Barras", "Mapa de calor", "Regresión Lineal", "Regresión Múltiple"
]
views_keys = ["view1", "view2", "view3", "view4", "view5", "view6", "view7"]

for i in range(7):
    with cols[i]:
        if st.button(labels[i]):
            st.session_state.view = views_keys[i]

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

def view_univariado():
    st.header("Análisis Univariado")
    st.sidebar.title("Panel de Control")
    st.sidebar.subheader("Explora los datos")

    if st.sidebar.checkbox("Mostrar dataset completo"):
        st.write(df)

    variable = st.sidebar.selectbox("Selecciona variable numérica", options=numeric_cols)
    fig = px.histogram(df, x=variable, nbins=20, title=f'Distribución de {variable}')
    st.plotly_chart(fig)
    st.markdown("Descripción estadística:")
    st.write(df[variable].describe())

def view_dispersion():
    st.header("Dispersión de Variables")
    x_var = st.sidebar.selectbox("Variable X", options=numeric_cols)
    y_var = st.sidebar.selectbox("Variable Y", options=numeric_cols)
    fig2 = px.scatter(df, x=x_var, y=y_var, title=f'Dispersión entre {x_var} y {y_var}')
    st.plotly_chart(fig2)

def view_pastel():
    st.subheader("🥧 Gráfico de pastel")

    cat_var = st.sidebar.selectbox("Variable Categórica", options=text_cols)
    num_var = st.sidebar.selectbox("Variable Numérica", options=numeric_cols)

    num_categorias = df[cat_var].nunique()

    if num_categorias > 30:
        st.warning(f"La variable '{cat_var}' tiene muchas categorías ({num_categorias}). Se mostrarán solo las 10 principales.")

    # Agrupar y filtrar las 10 categorías más frecuentes
    grouped_df = df.groupby(cat_var)[num_var].sum().sort_values(ascending=False).head(10).reset_index()

    fig3 = px.pie(grouped_df, names=cat_var, values=num_var,
                  title=f'Top 10 categorías de {cat_var} por {num_var}')
    st.plotly_chart(fig3)

def view_barras():
    st.header("Gráfico de Barras")
    cat_var = st.sidebar.selectbox("Variable Categórica", options=text_cols)
    num_var = st.sidebar.selectbox("Variable Numérica", options=numeric_cols)
    fig4 = px.bar(df, x=cat_var, y=num_var, title=f'{num_var} por {cat_var}')
    st.plotly_chart(fig4)

def view_heatmap():
    st.header("Mapa de Calor de Correlaciones")
    selected_vars = st.multiselect("Selecciona variables numéricas", options=numeric_cols, default=numeric_cols[:5])
    if len(selected_vars) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[selected_vars].corr(), annot=True, cmap="Purples", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Selecciona al menos dos variables.")

def view_regresion_lineal():
    st.header("Regresión Lineal Simple")
    x_var = st.sidebar.selectbox("Variable independiente (X)", options=numeric_cols)
    y_var = st.sidebar.selectbox("Variable dependiente (Y)", options=numeric_cols)

    X = df[[x_var]]
    y = df[y_var]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r = np.corrcoef(df[x_var], df[y_var])[0, 1]
    r_squared = model.score(X, y)

    fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", title=f'Regresión Lineal: {x_var} vs {y_var}')
    st.plotly_chart(fig)

    st.markdown(f"""
    **Ecuación:**  
    y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}  
    **R²:** {r_squared:.4f}  
    **r:** {r:.4f}
    """)

def view_regresion_multiple():
    st.header("Regresión Lineal Múltiple")
    y_var = st.sidebar.selectbox("Variable dependiente (Y)", options=numeric_cols)
    x_vars = st.sidebar.multiselect("Variables independientes (X)", options=[col for col in numeric_cols if col != y_var])

    if x_vars:
        X = df[x_vars]
        y = df[y_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r_squared = model.score(X_test, y_test)
        r = np.corrcoef(y_test, y_pred)[0, 1]

        st.markdown(f"**R² (Test):** {r_squared:.4f}")
        st.markdown(f"**r (Correlación):** {r:.4f}")

        coef_df = pd.DataFrame({"Variable": x_vars, "Coeficiente": model.coef_})
        st.write(coef_df)

        result_df = pd.DataFrame({"Real": y_test, "Predicho": y_pred}).reset_index(drop=True)
        fig = px.line(result_df, title="Comparación: Real vs Predicho")
        st.plotly_chart(fig)

# 4. Ejecutar la vista seleccionada
views = {
    "view1": view_univariado,
    "view2": view_dispersion,
    "view3": view_pastel,
    "view4": view_barras,
    "view5": view_heatmap,
    "view6": view_regresion_lineal,
    "view7": view_regresion_multiple,
}

views[st.session_state.view]()  # Ejecuta la función correspondiente

##############################################################################
#Menu desplegable de opciones de laa páginas seleccionadas
#View= st.selectbox(label= "view", options= ["view1", "view2", "view3", "view4", "view5", "view6", "view7"])

# VISTAS #
query_params = st.experimental_get_query_params()
view = query_params.get("view", ["view1", ])[0]



