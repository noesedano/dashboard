#Creamos el archivo de la APP en el interprete principal (Python)

##########################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_option_menu import option_menu
from funpymodeling.exploratory import freq_tbl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.set_page_config(layout="wide")
##########################################
#Definimos la instancia
@st.cache_resource


##########################################
#Creamos la funcion de carga de datos
def load_data():
    #Lectura del archivo csv
    df_chicago =pd.read_csv("Chicago.csv")
    #df_chicago = df_chicago.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1)

    df_mexico =pd.read_csv("Mexico.csv")
    #df_mexico = df_mexico.drop(['Unnamed: 0'], axis = 1)
    mexico_numeric_df = df_mexico.select_dtypes(['float', 'int'])
    mexico_text_df = df_mexico.select_dtypes(['object']) 

    #Selecciono las columnas tipo numericas del dataframe
    chicago_numeric_df = df_chicago.select_dtypes(['float', 'int'])   #Devuelve columnas
    numeric_cols = chicago_numeric_df.columns                         #Devuelve lista de columnas

    #Selecciono las columnas tipo texto del dataframe
    chicago_text_df = df_chicago.select_dtypes(['object'])   #Devuelve columnas
    text_cols = chicago_text_df.columns                         #Devuelve lista de columnas
    
    #Carga de Praga (PragaClean1)
    df_praga =pd.read_csv("PragaClean1.csv")
    praga_numeric_df = df_praga.select_dtypes(['float', 'int'])
    praga_text_df = df_praga.select_dtypes(['object']) 

    #Carga de  (buenos aires)
    df_buenosaires =pd.read_csv("BuenosA.csv")
    buenosaires_numeric_df = df_buenosaires.select_dtypes(['float', 'int'])
    buenosaires_text_df = df_buenosaires.select_dtypes(['object']) 

    return df_chicago, df_mexico, df_praga, df_buenosaires, numeric_cols, text_cols, chicago_numeric_df, chicago_text_df, mexico_numeric_df, mexico_text_df, praga_numeric_df, praga_text_df, buenosaires_numeric_df, buenosaires_text_df

##########################################
#Cargo los datos obtenidos de la funcion "load_data"
#Arg
df_chicago, df_mexico, df_praga, df_buenosaires, numeric_cols, text_cols, chicago_numeric_df, chicago_text_df, mexico_numeric_df, mexico_text_df, praga_numeric_df, praga_text_df, buenosaires_numeric_df, buenosaires_text_df = load_data()

##########################################
#Dashboard

#Navbar
View = option_menu(
    menu_title=None,  # Oculta el título
    options= ["Inicio", "Modelado explicativo", "Modelado predictivo"],
    icons=["house", "graph-up", "cpu"],  # Íconos de Bootstrap
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

#Logo
import base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_img_as_base64("img/airbnb.png")

##########################################
#Index

if View == "Inicio":
    st.title("Airbnb")
    st.write("Este dashboard presenta un Modelado explicativo usando un análisis univariado "
                        "de las variables categóricas más significativas y un Modelado predictivo usando "
                        "un análisis aplicando regresión lineal simple, regresión lineal multiple y regresión logistica,"
                        "esto haciendo uso de los datos propios de Airbnb acerca de las ciudades de Chicago Illinois, EU y Ciudad de México, MX.")
    img, title = st.columns([1, 7])
    left, right = st.columns([3, 1])
    img.image("img/airbnb.png", width=80)
    title.header("Acerca de Airbnb")
    left.subheader("¿Que es?")
    left.write("Airbnb es una plataforma digital que conecta a personas que desean alquilar su propiedad "
                            "(total o parcialmente) con viajeros que buscan alojamiento temporal. Fundada en 2008, "
                            "Airbnb ha transformado la industria del hospedaje, ofreciendo alternativas más flexibles y "
                            "personalizadas que los hoteles tradicionales.A través de su modelo de economía colaborativa, "
                            "permite que anfitriones publiquen espacios disponibles y que huéspedes puedan reservarlos de "
                            "forma segura, utilizando filtros como precio, ubicación, tipo de propiedad, calificaciones, y más.")
    right.image("img/airbnb.jpg", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                                    - Opera en más de 220 países y regiones.
                                    - Más de 4 millones de anfitriones en todo el mundo.
                                    - Más de 150 millones de usuarios han reservado a través de la plataforma.
                                    - Ofrece desde alojamientos económicos hasta opciones de lujo (Airbnb Luxe).
                                    """)
    img, title = st.columns([1, 7])
    left, right = st.columns([1, 3])
    img.image("img/mx.jpg", width=80)
    title.header("Acerca de México")
    right.subheader("¿Por qué México?")
    right.write("La Ciudad de México, capital del país y una de las urbes más grandes y vibrantes de América Latina, "
                                "es un centro cultural, histórico y económico de primer nivel. Su mezcla única de tradiciones prehispánicas, "
                                "arquitectura colonial y modernidad contemporánea atrae tanto a turistas como a viajeros de negocios de "
                                "todo el mundo.")
    right.write("En el contexto de Airbnb, la Ciudad de México representa un mercado urbano en constante expansión, con "
                                "una amplia variedad de alojamientos que van desde lofts modernos en colonias como Roma y Condesa hasta casas "
                                "tradicionales en Coyoacán o San Ángel. Su dinamismo turístico, la riqueza cultural y la constante realización de eventos "
                                "internacionales hacen de esta ciudad un caso clave para el análisis de comportamiento en plataformas de hospedaje.")
    left.image("img/mexico.jpg", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                                    - 📍 Ubicación: Centro del país, dentro del Valle de México.
                                    - 🌆 Población: Aproximadamente 9.2 millones de habitantes.
                                    - 🗺 Ubicación estratégica: Altiplano central, con conexión a rutas nacionales e internacionales.
                                    - ✈ Fácil acceso internacional: El Aeropuerto Internacional Benito Juárez es uno de los más transitados de América Latina, con conexiones a múltiples destinos globales.
                                    - 🏙 Principales atracciones:
                                        - Zócalo y Centro Histórico (Patrimonio de la Humanidad)
                                        - Museo Nacional de Antropología
                                        - Castillo de Chapultepec y su bosque
                                        - Coyoacán (hogar de Frida Kahlo)
                                        - Xochimilco y sus trajineras
                                    - 🍕 Gastronomía icónica:
                                        - Tacos al pastor, tamales, pozole
                                        - Alta cocina mexicana reconocida internacionalmente
                                        - Escena culinaria diversa en colonias como Polanco, Roma y Condesa
                                    """)
    img, title = st.columns([1, 7])
    left, right = st.columns([3, 1])
    img.image("img/usa.png", width=80)
    title.header("Acerca de Chicago")
    left.subheader("¿Por qué Chicago?")
    left.write("Chicago, ubicada en el estado de Illinois, es la tercera ciudad más grande de Estados Unidos y uno "
                        "de los destinos turísticos y culturales más importantes del país. Con una arquitectura emblemática, "
                        "una escena artística vibrante y una rica historia, la ciudad atrae a millones de visitantes cada año.")
    left.write("En el contexto de Airbnb, Chicago representa un mercado urbano dinámico con una gran diversidad "
                                "de alojamientos, desde apartamentos modernos en el centro hasta casas históricas en barrios "
                                "residenciales. Su perfil turístico, junto con eventos internacionales y zonas de alta demanda como "
                                "The Loop, Lincoln Park o Wicker Park, la convierten en un punto clave para el análisis de comportamiento "
                                "en plataformas de hospedaje.")
    right.image("img/chicago.jpg", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                                    - 📍 Ubicación: Estado de Illinois, Estados Unidos
                                    - 🌆 Población: Aproximadamente 2.7 millones de habitantes.
                                    - 🗺 Ubicación estratégica: A orillas del lago Míchigan, con vistas panorámicas y actividades acuáticas.
                                    - ✈ Fácil acceso internacional: El Aeropuerto O’Hare es uno de los más transitados del mundo, con vuelos a casi todos los continentes.
                                    - 🏙 Principales atracciones:
                                        - Millennium Park (con el famoso "Bean")
                                        - Willis Tower (Skydeck con piso de vidrio)
                                        - Art Institute of Chicago (uno de los mejores museos del mundo)
                                        - Riverwalk (paseo a lo largo del río Chicago)
                                        - Navy Pier (zona de entretenimiento junto al lago)
                                    - 🍕 Gastronomía icónica:
                                        - Pizza estilo Chicago (deep-dish)
                                        - Hot dogs "Chicago-style"
                                        - Gran oferta multicultural en barrios como Pilsen, Chinatown y Little Italy
                                    """)
    img, title = st.columns([1, 7])
    left, right = st.columns([1, 3])
    img.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
    title.header("Acerca de República Checa")
    right.subheader("¿Por qué República Checa?")
    right.write("La República Checa, con Praga como su capital y joya turística, es un destino europeo conocido por su historia medieval, "
                "arquitectura gótica y barroca, y su vibrante vida cultural. Praga es uno de los destinos más visitados de Europa Central, "
                "atrayendo a millones de turistas cada año por su belleza arquitectónica y ambiente acogedor.")
    right.write("En el contexto de Airbnb, la ciudad de Praga representa un mercado europeo consolidado y de alto valor turístico. "
                "Con una oferta diversa de alojamientos que van desde apartamentos en edificios históricos hasta modernos lofts junto al río Moldava, "
                "la capital checa permite estudiar cómo interactúan la demanda estacional, la regulación del alquiler a corto plazo y la oferta "
                "turística en un entorno urbano altamente competitivo.")
    left.image("https://media.istockphoto.com/id/1179665824/es/foto/praga-rep%C3%BAblica-checa.jpg?s=612x612&w=0&k=20&c=hK0fZY0NtPosI56JVkhiHFp4CwOp5mE-FdU3gaC3WFA=", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                - 📍 Ubicación: Europa Central, rodeada por Alemania, Austria, Eslovaquia y Polonia.
                - 🌆 Población: Alrededor de 10.7 millones en todo el país; más de 1.3 millones en Praga.
                - 🗺 Patrimonio cultural: Ciudad vieja de Praga declarada Patrimonio de la Humanidad por la UNESCO.
                - ✈ Acceso internacional: Aeropuerto Václav Havel conecta con destinos europeos, asiáticos y americanos.
                - 🏙 Principales atracciones:
                    - Castillo de Praga (uno de los más grandes del mundo)
                    - Puente de Carlos (icono gótico sobre el Moldava)
                    - Plaza de la Ciudad Vieja con el Reloj Astronómico
                    - Barrio Judío y sinagogas históricas
                    - Museo Nacional y la colina de Petřín
                - 🍺 Gastronomía y bebida:
                    - Cerveza checa mundialmente reconocida (Pilsner, Budvar)
                    - Platos típicos como goulash, svíčková, knedlíky
                    - Cafeterías bohemias y restaurantes de cocina internacional
                """)
    img, title = st.columns([1, 7])
    left, right = st.columns([3, 1])
    
    img.image("img/arg.jpg", width=80)
    title.header("Acerca de Buenos Aires")
    left.subheader("¿Por qué Buenos Aires?")
    left.write("Buenos Aires es la capital y ciudad más poblada de la República Argentina. Sus nombres oficiales son Ciudad de Buenos Aires o Ciudad Autónoma de Buenos Aires (CABA).​ También es llamada Capital Federal "
                "Está situada en la región centro-este del país, sobre la orilla sur del Río de la Plata"
                "Buenos Aires es una ciudad cosmopolita y un importante destino turístico.Su compleja infraestructura la convierte en una de las metrópolis de mayor importancia de América")
    left.write("En el contexto de Airbnb, tiene una relevancia considerable en el ecosistema de Airbnb en América Latina, tanto por volumen de oferta como por comportamiento de los usuarios."
                "Debido a las restricciones cambiarias en Argentina, Airbnb se convirtió en una forma de generar ingresos en moneda fuerte (USD) para propietarios")                           
    right.image("img/buenosa.jpg", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                - 📍 Conocida como la "París de Sudamérica" por su arquitectura, cultura y vida nocturna.
                - 🌆 Aeropuerto principal: Ezeiza (EZE), con vuelos directos a América, Europa y algunos destinos de Asia.
                - 🗺 Tango: Declarado Patrimonio Cultural Inmaterial por la UNESCO. Es común ver shows en vivo y clases.
                - ✈ Atrae más de 2 millones de turistas internacionales al año, siendo la ciudad más visitada de Sudamérica junto con Río de Janeiro
                - 🏙 Principales atracciones:
                    - Teatro Colón: Uno de los teatros líricos más importantes del mundo.
                    - Museos: Museo Nacional de Bellas Artes, MALBA (arte latinoamericano), Museo de Arte Moderno.
                    - Cafés históricos: Café Tortoni es el más emblemático.
                - 🍺 Gastronomía y bebida:
                    - Famosa por su carne vacuna, especialmente el asado argentino.
                    - Cultura de cafés y panaderías (facturas, medialunas).
                    - Creciente escena de restaurantes gourmet y cocina de autor.
                """)                            
  
    
    

                                    

                                    

##########################################
#Modelado explicativo

elif View == "Modelado explicativo":

    #Variable para tipo de variable a graficar
    if 'variable_type' not in st.session_state:
        st.session_state.variable_type = 'numeric'

    #Titulos y encabezados
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_base64}" width="70">
        <h1 style="margin-left: 10px; margin-bottom: 0;">Airbnb</h1>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("DASHBOARD")

    #Mostrar Dataset
    st.sidebar.subheader("ℹ Dataset")
    check_box = st.sidebar.checkbox(label = "Mostrar Dataset")
    if check_box:
        #Chicago
        st.subheader("Chicago dataset info:")
        st.write(df_chicago)
        st.write(df_chicago.columns)
        st.write(df_chicago.describe())
        #Mexico
        st.subheader("México dataset info:")
        st.write(df_mexico)
        st.write(df_mexico.columns)
        st.write(df_mexico.describe())
        #praga
        st.subheader("Praga dataset info:")
        st.write(df_praga)
        st.write(df_praga.columns)
        st.write(df_praga.describe())
         #buenos aires
        st.subheader("BuenosAires dataset info:")
        st.write(df_buenosaires)
        st.write(df_buenosaires.columns)
        st.write(df_buenosaires.describe())

    st.sidebar.header("⚙ Opciones")

    #Cambiar entre numericas y categoricas
    st.sidebar.subheader("🧬 Variables")
    button_type_variable = st.sidebar.button(label = "Cambiar tipo de variable")
    if button_type_variable:
        if st.session_state.variable_type == 'numeric':
            st.session_state.variable_type = 'categoric'
        else:
            st.session_state.variable_type = 'numeric'
    st.sidebar.write("Actual: " + st.session_state.variable_type)
    if st.session_state.variable_type == 'categoric':
        category_variable_selected = st.sidebar.selectbox(label = "Variables categoricas", options = text_cols)
        table = freq_tbl(df_chicago[category_variable_selected])
        table3 = freq_tbl(df_mexico[category_variable_selected])
        table6 = freq_tbl(df_praga[category_variable_selected])
        table9 = freq_tbl(df_buenosaires[category_variable_selected])
    else:
        numeric_variable_selected = st.sidebar.selectbox(label = "Variables numericas", options = numeric_cols)
        #Categorizar las variables numerica (Chicago)
        dataNumeric = df_chicago.copy()
        n = 8269
        Max = dataNumeric[numeric_variable_selected].max()
        Min = dataNumeric[numeric_variable_selected].min()
        R = Max - Min
        ni = max(5, min(12, round(1 + 3.32 * np.log10(n))))
        intervalos = np.linspace(Min, Max, ni + 1)
        categorias = [f"Intervalo de: {intervalos[i]:.2f} a {intervalos[i+1]:.2f}" for i in range(len(intervalos) - 1)]
        dataNumeric[numeric_variable_selected] = pd.cut(x = dataNumeric[numeric_variable_selected], bins = intervalos, labels = categorias)
        table = freq_tbl(dataNumeric[numeric_variable_selected])
        #Categorizar las variables numerica (Mexico)
        dataNumeric = df_mexico.copy()
        n = 26582
        Max = dataNumeric[numeric_variable_selected].max()
        Min = dataNumeric[numeric_variable_selected].min()
        R = Max - Min
        ni = max(5, min(12, round(1 + 3.32 * np.log10(n))))
        intervalos = np.linspace(Min, Max, ni + 1)
        categorias = [f"Intervalo de: {intervalos[i]:.2f} a {intervalos[i+1]:.2f}" for i in range(len(intervalos) - 1)]
        dataNumeric[numeric_variable_selected] = pd.cut(x = dataNumeric[numeric_variable_selected], bins = intervalos, labels = categorias)
        table3 = freq_tbl(dataNumeric[numeric_variable_selected])
        #Categorizar las variables numerica (praga)
        dataNumeric = df_praga.copy()
        n = 10108
        Max = dataNumeric[numeric_variable_selected].max()
        Min = dataNumeric[numeric_variable_selected].min()
        R = Max - Min
        ni = max(5, min(12, round(1 + 3.32 * np.log10(n))))
        intervalos = np.linspace(Min, Max, ni + 1)
        categorias = [f"Intervalo de: {intervalos[i]:.2f} a {intervalos[i+1]:.2f}" for i in range(len(intervalos) - 1)]
        dataNumeric[numeric_variable_selected] = pd.cut(x = dataNumeric[numeric_variable_selected], bins = intervalos, labels = categorias)
        table6 = freq_tbl(dataNumeric[numeric_variable_selected])
        #Categorizar las variables numerica (buenos aires)
        dataNumeric = df_buenosaires.copy()
        n = 35172
        Max = dataNumeric[numeric_variable_selected].max()
        Min = dataNumeric[numeric_variable_selected].min()
        R = Max - Min
        ni = max(5, min(12, round(1 + 3.32 * np.log10(n))))
        intervalos = np.linspace(Min, Max, ni + 1)
        categorias = [f"Intervalo de: {intervalos[i]:.2f} a {intervalos[i+1]:.2f}" for i in range(len(intervalos) - 1)]
        dataNumeric[numeric_variable_selected] = pd.cut(x = dataNumeric[numeric_variable_selected], bins = intervalos, labels = categorias)
        table9 = freq_tbl(dataNumeric[numeric_variable_selected])

    #Cambiar la frecuencia para los graficos
    st.sidebar.subheader("🔍 Filtro de frecuencia")
    frequency = st.sidebar.number_input("Frecuencia: " , min_value=0, max_value=None, value=0, step=None, format="%d")

    #Mostrar analisis univariado
    st.sidebar.subheader("🧪 Análisis univariado")
    check_box_analysis = st.sidebar.checkbox(label = "Mostrar analisis")
    if check_box_analysis:
        #Obtengo un analisis univariado de una variable en especifico
        if st.session_state.variable_type == 'categoric':
            st.header("Análisis univariado de: " + category_variable_selected)
        else:
            st.header("Análisis univariado de: " + numeric_variable_selected)
        #Chicago
        table2 = table[table['frequency'] > frequency]
        #Mexico
        table4 = table3[table3['frequency'] > frequency]
        #Praga
        table7 = table6[table6['frequency'] > frequency]
        #Argentina
        table10 = table9[table9['frequency'] > frequency]
        
        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.write(table2)
        left.image("img/mx.jpg", width=80)
        left.write(table4)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.write(table4)
        right.image("img/arg.jpg", width=80)
        right.write(table10)


    #Mostrar graficos
    st.sidebar.subheader("Graficos 📊")
    check_box_line = st.sidebar.checkbox(label = "📈 Grafico de lineas")
    check_box_bars = st.sidebar.checkbox(label = "📊 Grafico de barras")
    check_box_scatter = st.sidebar.checkbox(label = "🟢 Grafico de dispersion")
    check_box_area = st.sidebar.checkbox(label = "📉 Grafico de area")
    check_box_pie = st.sidebar.checkbox(label = "🥧 Grafico de pastel")
    check_box_box = st.sidebar.checkbox(label = "📦 Grafico de caja")

    #Grafica de lineas

    if  check_box_line or check_box_bars or check_box_scatter or check_box_area or check_box_pie:
        if st.session_state.variable_type == 'categoric':
            st.header("Graficos de: " + category_variable_selected)
        else: 
            st.header("Graficos de: " + numeric_variable_selected)

    if check_box_line:
        st.subheader("Line Plot")
        if st.session_state.variable_type == 'categoric':
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.line(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.line(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.line(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.line(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
        else:
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.line(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.line(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.line(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
             #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.line(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)
        

    if check_box_bars:
        st.subheader("Bar Plot")
        if st.session_state.variable_type == 'categoric':
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.bar(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", color= category_variable_selected, width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.bar(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", color= category_variable_selected, width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.bar(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", color= category_variable_selected, width = 1600, height =600)
            #buenos aires
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.bar(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", color= category_variable_selected, width = 1600, height =600)
        else:
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.bar(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", color= numeric_variable_selected, width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.bar(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", color= numeric_variable_selected, width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.bar(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", color= numeric_variable_selected, width = 1600, height =600)
            #Arg
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.bar(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", color= numeric_variable_selected, width = 1600, height =600)

        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        #Arg
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)

    if check_box_scatter:
        st.subheader("Scatter Plot")
        if st.session_state.variable_type == 'categoric':
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.scatter(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.scatter(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.scatter(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.scatter(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
        else:
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.scatter(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.scatter(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.scatter(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.scatter(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)

        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)


    if check_box_area:
        st.subheader("Area Plot")
        if st.session_state.variable_type == 'categoric':
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.area(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.area(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.area(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.area(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
        else:
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.area(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.area(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.area(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)
            #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.area(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)

        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)

    if check_box_pie:
        st.subheader("Pie Plot")
        if st.session_state.variable_type == 'categoric':
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.pie(data_frame = Filtro, names = category_variable_selected, 
            values = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.pie(data_frame = Filtro, names = category_variable_selected, 
            values = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.pie(data_frame = Filtro, names = category_variable_selected, 
            values = "frequency", width = 1600, height =600)
            #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.pie(data_frame = Filtro, names = category_variable_selected, 
            values = "frequency", width = 1600, height =600)
        else:
            #Chicago
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.pie(data_frame = Filtro, names = numeric_variable_selected, 
            values = "frequency", width = 1600, height =600)
            #Mexico
            table4 = table3.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table4[table4['frequency'] > frequency]
            figure2 = px.pie(data_frame = Filtro, names = numeric_variable_selected, 
            values = "frequency", width = 1600, height =600)
            #Praga
            table7 = table6.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table7[table7['frequency'] > frequency]
            figure3 = px.pie(data_frame = Filtro, names = numeric_variable_selected, 
            values = "frequency", width = 1600, height =600)
            #argentina
            table10 = table9.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table10[table10['frequency'] > frequency]
            figure4 = px.pie(data_frame = Filtro, names = numeric_variable_selected, 
            values = "frequency", width = 1600, height =600)


        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)

    if check_box_box:
        st.subheader("Box Plot")
        if st.session_state.variable_type == 'categoric':
            #Chicago
            figure1 = px.box(df_chicago, x=category_variable_selected, orientation='h')
            #Mexico
            figure2 = px.box(df_mexico, x=category_variable_selected, orientation='h')
            #Praga
            figure3 = px.box(df_praga, x=category_variable_selected, orientation='h')
            #argentina
            figure4 = px.box(df_buenosaires, x=category_variable_selected, orientation='h')
        else:
            #Chicago
            figure1 = px.box(df_chicago, x=numeric_variable_selected, orientation='h')
            #Mexico
            figure2 = px.box(df_mexico, x=numeric_variable_selected, orientation='h')
            #Praga
            figure3 = px.box(df_praga, x=numeric_variable_selected, orientation='h')
            #argentina
            figure4 = px.box(df_buenosaires, x=numeric_variable_selected, orientation='h')

        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)


##########################################
#Contenido de la vista 2
elif View == "Modelado predictivo":

    #Variable para tipo de variable a graficar
    if 'variable_type' not in st.session_state:
        st.session_state.variable_type = 'numeric'

    if st.session_state.variable_type == 'categoric':
            for col in chicago_text_df:
                #Chicago
                frequencies = chicago_text_df[col].value_counts()
                chicago_text_df[col] = chicago_text_df[col].map(frequencies)
                #Mexico
                frequencies2 = mexico_text_df[col].value_counts()
                mexico_text_df[col] = mexico_text_df[col].map(frequencies2)
                #Praga
                frequencies3 = praga_text_df[col].value_counts()
                praga_text_df[col] = praga_text_df[col].map(frequencies3)
                #Arg
                frequencies4 = buenosaires_text_df[col].value_counts()
                buenosaires_text_df[col] = buenosaires_text_df[col].map(frequencies4)

    #Titulos y encabezados 
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_base64}" width="70">
        <h1 style="margin-left: 10px; margin-bottom: 0;">Airbnb</h1>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("DASHBOARD")

    st.sidebar.subheader("ℹ Dataset")
    check_box = st.sidebar.checkbox(label = "Mostrar Dataset")
    st.sidebar.header("⚙ Opciones")

    #Cambiar entre numericas y categoricas
    st.sidebar.subheader("🧬 Variables")
    button_type_variable = st.sidebar.button(label = "Cambiar tipo de variable")
    if button_type_variable:
        if st.session_state.variable_type == 'numeric':
            st.session_state.variable_type = 'categoric'
        else:
            st.session_state.variable_type = 'numeric'
    st.sidebar.write("Actual: " + st.session_state.variable_type)

    check_box_heatmap = st.sidebar.checkbox(label = "🌡 Mapa de calor")

    st.sidebar.subheader("➡📉 Regresion lineal simple")
    if st.session_state.variable_type == 'numeric':
        x_options = numeric_cols
        y_options = numeric_cols
    else:
        x_options = text_cols
        y_options = text_cols
    x_selected_simple = st.sidebar.selectbox(label = "x", options = x_options, key = "simple_x")
    y_selected_simple = st.sidebar.selectbox(label = "y", options = y_options, key = "simple_y")
    check_box_scatter_simple = st.sidebar.checkbox(label = "🟢 Diagrama de dispersion", key = "simple_scatter")
    check_box_info_simple = st.sidebar.checkbox(label = "ℹ Informacion del modelo", key = "simple_info")

    st.sidebar.subheader("➡📊 Regresion lineal multiple")
    x_selected_multi = st.sidebar.multiselect(label = "x", options = x_options, key = "multi_x")
    y_selected_multi = st.sidebar.selectbox(label = "y", options = y_options, key = "multi_y")
    check_box_scatter_multi = st.sidebar.checkbox(label = "🟢 Diagrama de dispersion", key = "multi_scatter")
    check_box_info_multi = st.sidebar.checkbox(label = "ℹ Informacion del modelo", key = "multi_info")

    st.sidebar.subheader("📊🔑 Regresion logistica")
    x_selected_log = st.sidebar.multiselect(label = "x", options = numeric_cols, key = "log_x")
    y_selected_log = st.sidebar.selectbox(label = "y", options = text_cols, key = "log_y")
    dichotomous_column = df_chicago[y_selected_log]
    unique_categories = dichotomous_column.unique()
    val_selected_log = st.sidebar.selectbox(label = "Valor a predecir", options = unique_categories)
    check_box_matriz = st.sidebar.checkbox(label = "❓🔲 Matriz de confusion")
    check_box_info_log = st.sidebar.checkbox(label = "ℹ Informacion del modelo", key = "log_info")

    #Mostrar Dataset
    if check_box:
        #Chicago
        st.subheader("Chicago dataset info:")
        st.write(df_chicago)
        st.write(df_chicago.columns)
        st.write(df_chicago.describe())
        #Mexico
        st.subheader("México dataset info:")
        st.write(df_mexico)
        st.write(df_mexico.columns)
        st.write(df_mexico.describe())
        #Praga
        st.subheader("Praga dataset info:")
        st.write(df_praga)
        st.write(df_praga.columns)
        st.write(df_praga.describe())
        #Argentina
        st.subheader("Buenos Aires dataset info:")
        st.write(df_buenosaires)
        st.write(df_buenosaires.columns)
        st.write(df_buenosaires.describe())

    #Mapa de calor
    if check_box_heatmap:
        st.subheader("Heatmap")
        if st.session_state.variable_type == 'numeric':
            #Chicago
            correlation = abs(chicago_numeric_df.corr())
            #Mexico
            correlation2 = abs(mexico_numeric_df.corr())
            #Praga
            correlation3 = abs(praga_numeric_df.corr())
             #Mexico
            correlation4 = abs(buenosaires_numeric_df.corr())
        else:
            #Chicago
            correlation = abs(chicago_text_df.corr())
            #Mexico
            correlation2 = abs(mexico_text_df.corr())
            #Praga
            correlation3 = abs(praga_text_df.corr())
            #argentina
            correlation4 = abs(buenosaires_text_df.corr())

        #Chicago
        figure1 = px.imshow(
            correlation,
            text_auto=True,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        #Mexico
        figure2 = px.imshow(
            correlation2,
            text_auto=True,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        #Praga
        figure3 = px.imshow(
            correlation3,
            text_auto=True,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        #argentina
        figure4 = px.imshow(
            correlation4,
            text_auto=True,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)


    #Regresion lineal simple
    if check_box_scatter_simple or check_box_info_simple:
        st.header("Regresion Lineal Simple")
        if st.session_state.variable_type == 'numeric':
            #Chicago
            Vars_Indep_chicago = chicago_numeric_df[[x_selected_simple]] 
            Var_Dep_chicago = chicago_numeric_df[y_selected_simple]
            df5  = chicago_numeric_df
            #Mexico
            Vars_Indep_mexico = mexico_numeric_df[[x_selected_simple]] 
            Var_Dep_mexico = mexico_numeric_df[y_selected_simple]
            df6  = mexico_numeric_df
            #Praga
            Vars_Indep_praga = praga_numeric_df[[x_selected_simple]] 
            Var_Dep_praga = praga_numeric_df[y_selected_simple]
            df6  = praga_numeric_df
            #argentina
            Vars_Indep_buenosaires = buenosaires_numeric_df[[x_selected_simple]] 
            Var_Dep_buenosaires = buenosaires_numeric_df[y_selected_simple]
            df8  = buenosaires_numeric_df
        else:
            #Chicago
            Vars_Indep_chicago = chicago_text_df[[x_selected_simple]] 
            Var_Dep_chicago = chicago_text_df[y_selected_simple]
            df5 = chicago_text_df
            #Mexico
            Vars_Indep_mexico = mexico_text_df[[x_selected_simple]] 
            Var_Dep_mexico = mexico_text_df[y_selected_simple]
            df6 = mexico_text_df
            #Praga
            Vars_Indep_praga = praga_text_df[[x_selected_simple]] 
            Var_Dep_praga = praga_text_df[y_selected_simple]
            df7 = praga_text_df
            #argentina
            Vars_Indep_buenosaires = buenosaires_text_df[[x_selected_simple]] 
            Var_Dep_buenosaires = buenosaires_text_df[y_selected_simple]
            df8 = buenosaires_text_df
        #Chicago
        model_chicago = LinearRegression()
        model_chicago.fit(X = Vars_Indep_chicago, y = Var_Dep_chicago)
        y_pred = model_chicago.predict(X = df5[[x_selected_simple]])
        df_simple_chicago = df5.copy()
        df_simple_chicago.insert(0, 'Predicciones', y_pred)
        #Mexico
        model_mexico = LinearRegression()
        model_mexico.fit(X = Vars_Indep_mexico, y = Var_Dep_mexico)
        y_pred = model_mexico.predict(X = df6[[x_selected_simple]])
        df_simple_mexico = df6.copy()
        df_simple_mexico.insert(0, 'Predicciones', y_pred)
        #Praga
        model_praga = LinearRegression()
        model_praga.fit(X = Vars_Indep_praga, y = Var_Dep_praga)
        y_pred = model_praga.predict(X = df6[[x_selected_simple]])
        df_simple_praga = df6.copy()
        df_simple_praga.insert(0, 'Predicciones', y_pred)
        #argentina
        model_buenosaires = LinearRegression()
        model_buenosaires.fit(X = Vars_Indep_buenosaires, y = Var_Dep_buenosaires)
        y_pred = model_buenosaires.predict(X = df8[[x_selected_simple]])
        df_simple_buenosaires = df8.copy()
        df_simple_buenosaires.insert(0, 'Predicciones', y_pred)



    #Diagramas de dispersion
    if check_box_scatter_simple:
        st.subheader("Scatter Plot")
        #Chicago
        df_long = pd.melt(df_simple_chicago, id_vars=x_selected_simple, value_vars=[y_selected_simple, "Predicciones"],
                  var_name="Tipo", value_name="Valor")
        figure1 = px.scatter(df_long, x=x_selected_simple, y="Valor", color="Tipo", color_discrete_map={
                "Predicciones": "red"
            }, width = 1600, height =600)
        #Mexico
        df_long = pd.melt(df_simple_mexico, id_vars=x_selected_simple, value_vars=[y_selected_simple, "Predicciones"],
                  var_name="Tipo", value_name="Valor")
        figure2 = px.scatter(df_long, x=x_selected_simple, y="Valor", color="Tipo", color_discrete_map={
                "Predicciones": "red"
            }, width = 1600, height =600)
        #Praga
        df_long = pd.melt(df_simple_praga, id_vars=x_selected_simple, value_vars=[y_selected_simple, "Predicciones"],
                  var_name="Tipo", value_name="Valor")
        figure3 = px.scatter(df_long, x=x_selected_simple, y="Valor", color="Tipo", color_discrete_map={
                "Predicciones": "red"
            }, width = 1600, height =600)
        #argentina
        df_long = pd.melt(df_simple_buenosaires, id_vars=x_selected_simple, value_vars=[y_selected_simple, "Predicciones"],
                  var_name="Tipo", value_name="Valor")
        figure4 = px.scatter(df_long, x=x_selected_simple, y="Valor", color="Tipo", color_discrete_map={
                "Predicciones": "red"
            }, width = 1600, height =600)
        left, right = st.columns([1, 1])
        right.image("img/usa.png", width=80)
        right.plotly_chart(figure1)
        left.image("img/mx.jpg", width=80)
        left.plotly_chart(figure2)
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.plotly_chart(figure3)
        right.image("img/arg.jpg", width=80)
        right.plotly_chart(figure4)
        

    #Mostrar info del modelo
    if check_box_info_simple:
        st.subheader("Model info: " + " " + y_selected_simple + " vs " + x_selected_simple)
        left, right = st.columns([1, 1])
        #Chicago
        coef_Deter = model_chicago.score(X = Vars_Indep_chicago, y = Var_Dep_chicago)
        coef_Correl = np.sqrt(coef_Deter)
        a = model_chicago.coef_[0]
        b = model_chicago.intercept_
        right.image("img/usa.png", width=80)
        right.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
        right.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
        right.write(f"Modelo matemático: y = {a:.4f}x + {b:.4f}")
        #Mexico
        coef_Deter = model_mexico.score(X = Vars_Indep_mexico, y = Var_Dep_mexico)
        coef_Correl = np.sqrt(coef_Deter)
        a = model_mexico.coef_[0]
        b = model_mexico.intercept_
        left.image("img/mx.jpg", width=80)
        left.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
        left.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
        left.write(f"Modelo matemático: y = {a:.4f}x + {b:.4f}")
        #Praga
        coef_Deter = model_praga.score(X = Vars_Indep_praga, y = Var_Dep_praga)
        coef_Correl = np.sqrt(coef_Deter)
        a = model_praga.coef_[0]
        b = model_praga.intercept_
        left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
        left.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
        left.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
        left.write(f"Modelo matemático: y = {a:.4f}x + {b:.4f}")
        #argentina
        coef_Deter = model_buenosaires.score(X = Vars_Indep_buenosaires, y = Var_Dep_buenosaires)
        coef_Correl = np.sqrt(coef_Deter)
        a = model_buenosaires.coef_[0]
        b = model_buenosaires.intercept_
        right.image("img/arg.jpg", width=80)
        right.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
        right.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
        right.write(f"Modelo matemático: y = {a:.4f}x + {b:.4f}")

    #Regresion lineal multiple
    if check_box_scatter_multi or check_box_info_multi:
        st.header("Regresion Lineal Multiple")
        if x_selected_multi:
            if st.session_state.variable_type == 'numeric':
                #Chicago
                Vars_Indep = chicago_numeric_df[x_selected_multi] 
                Var_Dep = chicago_numeric_df[y_selected_multi]
                df5 = chicago_numeric_df
                #Mexico
                Vars_Indep2 = mexico_numeric_df[x_selected_multi]
                Var_Dep2 = mexico_numeric_df[y_selected_multi]
                df6 = mexico_numeric_df
                #Praga
                Vars_Indep3 = praga_numeric_df[x_selected_multi]
                Var_Dep3 = praga_numeric_df[y_selected_multi]
                df7 = praga_numeric_df
                #buenos aires
                Vars_Indep4 = buenosaires_numeric_df[x_selected_multi] 
                Var_Dep4 = buenosaires_numeric_df[y_selected_multi]
                df8 = buenosaires_numeric_df
            else:
                #Chicago
                Vars_Indep = chicago_text_df[x_selected_multi] 
                Var_Dep = chicago_text_df[y_selected_multi]
                df5 = chicago_text_df
                #Mexico
                Vars_Indep2 = mexico_text_df[x_selected_multi]
                Var_Dep2 = mexico_text_df[y_selected_multi]
                df6 = mexico_text_df
                #Praga
                Vars_Indep3 = praga_text_df[x_selected_multi]
                Var_Dep3 = praga_text_df[y_selected_multi]
                df7 = praga_text_df
                #argentina
                Vars_Indep4 = buenosaires_text_df[x_selected_multi] 
                Var_Dep4 = buenosaires_text_df[y_selected_multi]
                df8 = buenosaires_text_df
            #Chicago
            model_chicago = LinearRegression()
            model_chicago.fit(X = Vars_Indep, y = Var_Dep)
            y_pred = model_chicago.predict(X = df5[x_selected_multi])
            df_multi = df5.copy()
            df_multi.insert(0, 'Predicciones', y_pred)
            #Mexico
            model_mexico = LinearRegression()
            model_mexico.fit(X = Vars_Indep2, y = Var_Dep2)
            y_pred = model_mexico.predict(X = df6[x_selected_multi])
            df_multi2 = df6.copy()
            df_multi2.insert(0, 'Predicciones', y_pred)
            #Praga
            model_praga = LinearRegression()
            model_praga.fit(X = Vars_Indep3, y = Var_Dep3)
            y_pred = model_praga.predict(X = df7[x_selected_multi])
            df_multi3 = df7.copy()
            df_multi3.insert(0, 'Predicciones', y_pred)
            #argentina
            model_buenosaires = LinearRegression()
            model_buenosaires.fit(X = Vars_Indep4, y = Var_Dep4)
            y_pred = model_buenosaires.predict(X = df8[x_selected_multi])
            df_multi4 = df8.copy()
            df_multi4.insert(0, 'Predicciones', y_pred)
        else:
            st.write("Selecione alguna variable x")

    #Diagrama de dispersion
    if check_box_scatter_multi:
        st.subheader("Scatter Plot")
        if x_selected_multi:
            #Chicago
            df_long = pd.melt(df_multi, id_vars=x_selected_multi, value_vars=[y_selected_multi, "Predicciones"],
                    var_name="Tipo", value_name="Valor")
            figure1 = px.scatter(df_long, x=x_selected_multi, y="Valor", color="Tipo", color_discrete_map={
                    "Predicciones": "red"
                }, width = 1600, height =600)
            #Mexico
            df_long = pd.melt(df_multi2, id_vars=x_selected_multi, value_vars=[y_selected_multi, "Predicciones"],
                    var_name="Tipo", value_name="Valor")
            figure2 = px.scatter(df_long, x=x_selected_multi, y="Valor", color="Tipo", color_discrete_map={
                    "Predicciones": "red"
                }, width = 1600, height =600)
            #Praga
            df_long = pd.melt(df_multi3, id_vars=x_selected_multi, value_vars=[y_selected_multi, "Predicciones"],
                    var_name="Tipo", value_name="Valor")
            figure3 = px.scatter(df_long, x=x_selected_multi, y="Valor", color="Tipo", color_discrete_map={
                    "Predicciones": "red"
                }, width = 1600, height =600)
                #Chicago
            df_long = pd.melt(df_multi4, id_vars=x_selected_multi, value_vars=[y_selected_multi, "Predicciones"],
                    var_name="Tipo", value_name="Valor")
            figure4 = px.scatter(df_long, x=x_selected_multi, y="Valor", color="Tipo", color_discrete_map={
                    "Predicciones": "red"
                }, width = 1600, height =600)

            left, right = st.columns([1, 1])
            right.image("img/usa.png", width=80)
            right.plotly_chart(figure1)
            left.image("img/mx.jpg", width=80)
            left.plotly_chart(figure2)
            left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
            left.plotly_chart(figure3)
            right.image("img/arg.jpg", width=80)
            right.plotly_chart(figure4)
    
    #Mostrar info del modelo
    if check_box_info_multi:
        if x_selected_multi:
            st.subheader(f"Model info: {y_selected_multi} vs {x_selected_multi}")
            left, right = st.columns([1, 1])
            #Chicago
            coef_Deter = model_chicago.score(X = Vars_Indep, y = Var_Dep)
            coef_Correl = np.sqrt(coef_Deter)
            a = model_chicago.coef_
            b = model_chicago.intercept_
            left.image("img/mx.jpg", width=80)
            left.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
            left.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
            model_math = "y = " + f"{b:.4f}"
            for i, coef in enumerate(a):
                model_math += f" + ({coef:.4f}) * {Vars_Indep.columns[i]}"
            left.write(f"Modelo matemático: {model_math}")
            #Mexico
            coef_Deter = model_mexico.score(X = Vars_Indep2, y = Var_Dep2)
            coef_Correl = np.sqrt(coef_Deter)
            a = model_mexico.coef_
            b = model_mexico.intercept_
            right.image("img/usa.png", width=80)
            right.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
            right.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
            model_math = "y = " + f"{b:.4f}"
            for i, coef in enumerate(a):
                model_math += f" + ({coef:.4f}) * {Vars_Indep.columns[i]}"
            right.write(f"Modelo matemático: {model_math}")
            #Praga
            coef_Deter = model_praga.score(X = Vars_Indep3, y = Var_Dep3)
            coef_Correl = np.sqrt(coef_Deter)
            a = model_praga.coef_
            b = model_praga.intercept_
            left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
            left.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
            left.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
            model_math = "y = " + f"{b:.4f}"
            for i, coef in enumerate(a):
                model_math += f" + ({coef:.4f}) * {Vars_Indep.columns[i]}"
            left.write(f"Modelo matemático: {model_math}")
            #argentina
            coef_Deter = model_buenosaires.score(X = Vars_Indep4, y = Var_Dep4)
            coef_Correl = np.sqrt(coef_Deter)
            a = model_buenosaires.coef_
            b = model_buenosaires.intercept_
            right.image("img/arg.jpg", width=80)
            right.write(f"R (Indice de correlacion) =  {coef_Correl:.4f}")
            right.write(f"R^2 (Indice de determinacion) = {coef_Deter:.4f}")
            model_math = "y = " + f"{b:.4f}"
            for i, coef in enumerate(a):
                model_math += f" + ({coef:.4f}) * {Vars_Indep.columns[i]}"
            right.write(f"Modelo matemático: {model_math}")

    #Regresion logistica
    if check_box_matriz or check_box_info_log:
        st.header("Regresion Logistica")
        if x_selected_log:
            #Chicago
            df2 = df_chicago.copy()
            df2[y_selected_log] =df2[y_selected_log].mask(df2[y_selected_log] != val_selected_log, "Other Value")
            Vars_Indep_chicago = df2[x_selected_log]
            Var_Dep_chicago = df2[y_selected_log]
            X = Vars_Indep_chicago
            y = Var_Dep_chicago
            X_train, X_test, y_train, y_test =train_test_split(X, y, test_size= 0.3, random_state=None)
            escalar = StandardScaler()
            X_train = escalar.fit_transform(X_train)
            X_test = escalar.transform(X_test)
            algoritmo = LogisticRegression()
            algoritmo.fit(X_train, y_train)
            y_pred = algoritmo.predict(X_test)
            matriz_chicago = confusion_matrix(y_test, y_pred, labels=["Other Value", val_selected_log])
            precision_chicago = precision_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            precision_other_chicago = precision_score(y_test, y_pred, average="binary", pos_label="Other Value")
            exactitud_chicago = accuracy_score(y_test, y_pred)
            sensibilidad_chicago = recall_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            sensibilidad_other_chicago = recall_score(y_test, y_pred, average="binary", pos_label="Other Value")
            #Mexico
            df2 = df_mexico.copy()
            df2[y_selected_log] =df2[y_selected_log].mask(df2[y_selected_log] != val_selected_log, "Other Value")
            Vars_Indep_mexico = df2[x_selected_log]
            Var_Dep_mexico = df2[y_selected_log]
            X = Vars_Indep_mexico
            y = Var_Dep_mexico
            X_train, X_test, y_train, y_test =train_test_split(X, y, test_size= 0.3, random_state=None)
            escalar = StandardScaler()
            X_train = escalar.fit_transform(X_train)
            X_test = escalar.transform(X_test)
            algoritmo_chicago = LogisticRegression()
            algoritmo_chicago.fit(X_train, y_train)
            y_pred = algoritmo_chicago.predict(X_test)
            matriz_mexico = confusion_matrix(y_test, y_pred, labels=["Other Value", val_selected_log])
            precision_mexico = precision_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            precision_other_mexico = precision_score(y_test, y_pred, average="binary", pos_label="Other Value")
            exactitud_mexico = accuracy_score(y_test, y_pred)
            sensibilidad_mexico = recall_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            sensibilidad_other_mexico = recall_score(y_test, y_pred, average="binary", pos_label="Other Value")
            #Praga
            df3 = df_praga.copy()
            df3[y_selected_log] =df3[y_selected_log].mask(df3[y_selected_log] != val_selected_log, "Other Value")
            Vars_Indep_praga = df3[x_selected_log]
            Var_Dep_praga = df3[y_selected_log]
            X = Vars_Indep_praga
            y = Var_Dep_praga
            X_train, X_test, y_train, y_test =train_test_split(X, y, test_size= 0.3, random_state=None)
            escalar = StandardScaler()
            X_train = escalar.fit_transform(X_train)
            X_test = escalar.transform(X_test)
            algoritmo_chicago = LogisticRegression()
            algoritmo_chicago.fit(X_train, y_train)
            y_pred = algoritmo_chicago.predict(X_test)
            matriz_praga = confusion_matrix(y_test, y_pred, labels=["Other Value", val_selected_log])
            precision_praga = precision_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            precision_other_praga = precision_score(y_test, y_pred, average="binary", pos_label="Other Value")
            exactitud_praga = accuracy_score(y_test, y_pred)
            sensibilidad_praga = recall_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            sensibilidad_other_praga = recall_score(y_test, y_pred, average="binary", pos_label="Other Value")

            #argentina
            df2 = df_buenosaires.copy()
            df2[y_selected_log] =df2[y_selected_log].mask(df2[y_selected_log] != val_selected_log, "Other Value")
            Vars_Indep_buenosaires = df2[x_selected_log]
            Var_Dep_buenosaires = df2[y_selected_log]
            X = Vars_Indep_buenosaires
            y = Var_Dep_buenosaires
            X_train, X_test, y_train, y_test =train_test_split(X, y, test_size= 0.3, random_state=None)
            escalar = StandardScaler()
            X_train = escalar.fit_transform(X_train)
            X_test = escalar.transform(X_test)
            algoritmo_chicago = LogisticRegression()
            algoritmo_chicago.fit(X_train, y_train)
            y_pred = algoritmo_chicago.predict(X_test)
            matriz_buenosaires = confusion_matrix(y_test, y_pred, labels=["Other Value", val_selected_log])
            precision_buenosaires = precision_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            precision_other_buenosaires = precision_score(y_test, y_pred, average="binary", pos_label="Other Value")
            exactitud_buenosaires = accuracy_score(y_test, y_pred)
            sensibilidad_buenosaires = recall_score(y_test, y_pred, average="binary", pos_label=val_selected_log)
            sensibilidad_other_buenosaires = recall_score(y_test, y_pred, average="binary", pos_label="Other Value")

        else:
            st.write("Selecione alguna variable x")

    if check_box_matriz:
        st.subheader('Matriz de Confusion')
        if x_selected_log:
            z = [[1, 0],
                [4, 1]]
            # Colores fijos por índice
            colorscale = [
                [0.0, '#64db4f'],
                [0.25, '#64db4f'],
                [0.25, '#e04040'],
                [0.5, '#e04040'],
                [0.5, '#e04040'],
                [0.75, '#64db4f'],
                [0.75, '#64db4f'],
                [1.0, '#64db4f']
            ]
            #Chicago
            # Anotaciones reales de tu matriz
            text = [[str(matriz_chicago[1][0]), str(matriz_chicago[0][0])],
                    [str(matriz_chicago[1][1]), str(matriz_chicago[0][1])]]
            # Crear figura tipo heatmap con anotaciones
            figure1 = go.Figure(data=go.Heatmap(
                z=z,
                x = ['Real Positivo', 'Real Negativo'],
                y = ['Pred. Negativo', 'Pred. Positivo'],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale=colorscale,
                showscale=False,
                hoverinfo="x+y+text"
            ))
            figure1.update_layout(
                xaxis_title='Valor Real',
                yaxis_title='Valor de Predicción'
            )
            #Mexico
            # Anotaciones reales de tu matriz
            text = [[str(matriz_mexico[1][0]), str(matriz_mexico[0][0])],
                    [str(matriz_mexico[1][1]), str(matriz_mexico[0][1])]]
            # Crear figura tipo heatmap con anotaciones
            figure2 = go.Figure(data=go.Heatmap(
                z=z,
                x = ['Real Positivo', 'Real Negativo'],
                y = ['Pred. Negativo', 'Pred. Positivo'],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale=colorscale,
                showscale=False,
                hoverinfo="x+y+text"
            ))
            figure2.update_layout(
                xaxis_title='Valor Real',
                yaxis_title='Valor de Predicción'
            )
            #praga
            # Anotaciones reales de tu matriz
            text = [[str(matriz_praga[1][0]), str(matriz_praga[0][0])],
                    [str(matriz_praga[1][1]), str(matriz_praga[0][1])]]
            # Crear figura tipo heatmap con anotaciones
            figure3 = go.Figure(data=go.Heatmap(
                z=z,
                x = ['Real Positivo', 'Real Negativo'],
                y = ['Pred. Negativo', 'Pred. Positivo'],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale=colorscale,
                showscale=False,
                hoverinfo="x+y+text"
            ))
            figure3.update_layout(
                xaxis_title='Valor Real',
                yaxis_title='Valor de Predicción'
            )
            #argentina
            # Anotaciones reales de la matriz
            text = [[str(matriz_buenosaires[1][0]), str(matriz_buenosaires[0][0])],
                    [str(matriz_buenosaires[1][1]), str(matriz_buenosaires[0][1])]]
            # Crear figura tipo heatmap con anotaciones
            figure4 = go.Figure(data=go.Heatmap(
                z=z,
                x = ['Real Positivo', 'Real Negativo'],
                y = ['Pred. Negativo', 'Pred. Positivo'],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale=colorscale,
                showscale=False,
                hoverinfo="x+y+text"
            ))
            figure4.update_layout(
                xaxis_title='Valor Real',
                yaxis_title='Valor de Predicción'
            )
            left, right = st.columns([1, 1])
            right.image("img/usa.png", width=80)
            right.plotly_chart(figure1)
            left.image("img/mx.jpg", width=80)
            left.plotly_chart(figure2)
            left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
            left.plotly_chart(figure3)
            right.image("img/arg.jpg", width=80)
            right.plotly_chart(figure4)
    
    if check_box_info_log:
        st.subheader(f"Model info: {y_selected_log} vs {x_selected_log}")
        if x_selected_log:
            left, right = st.columns([1, 1])
            #Mexico
            left.image("img/mx.jpg", width=80)
            left.write(f"Exactitud del modelo: { exactitud_mexico:.4f}")
            left.write(f"*Valor: {val_selected_log}*")
            left.write(f"Precision del modelo: {precision_mexico:.4f}")
            left.write(f"Sensibilidad del modelo: {sensibilidad_mexico:.4f}")
            left.write(f"*Valor: Otro*")
            left.write(f"Precision del modelo: {precision_other_mexico:.4f}")
            left.write(f"Sensibilidad del modelo: {sensibilidad_other_mexico:.4f}")
            #Chicago
            right.image("img/usa.png", width=80)
            right.write(f"Exactitud del modelo: { exactitud_chicago:.4f}")
            right.write(f"*Valor: {val_selected_log}*")
            right.write(f"Precision del modelo: {precision_chicago:.4f}")
            right.write(f"Sensibilidad del modelo: {sensibilidad_chicago:.4f}")
            right.write(f"*Valor: Otro*")
            right.write(f"Precision del modelo: {precision_other_chicago:.4f}")
            right.write(f"Sensibilidad del modelo: {sensibilidad_other_chicago:.4f}")
            #Praga
            left.image("https://upload.wikimedia.org/wikipedia/commons/c/cb/Flag_of_the_Czech_Republic.svg", width=80)
            left.write(f"Exactitud del modelo: { exactitud_praga:.4f}")
            left.write(f"*Valor: {val_selected_log}*")
            left.write(f"Precision del modelo: {precision_praga:.4f}")
            left.write(f"Sensibilidad del modelo: {sensibilidad_praga:.4f}")
            left.write(f"*Valor: Otro*")
            left.write(f"Precision del modelo: {precision_other_praga:.4f}")
            left.write(f"Sensibilidad del modelo: {sensibilidad_other_praga:.4f}")
            #argentina
            right.image("img/arg.jpg", width=80)
            right.write(f"Exactitud del modelo: { exactitud_buenosaires:.4f}")
            right.write(f"*Valor: {val_selected_log}*")
            right.write(f"Precision del modelo: {precision_buenosaires:.4f}")
            right.write(f"Sensibilidad del modelo: {sensibilidad_buenosaires:.4f}")
            right.write(f"*Valor: Otro*")
            right.write(f"Precision del modelo: {precision_other_buenosaires:.4f}")
            right.write(f"Sensibilidad del modelo: {sensibilidad_other_buenosaires:.4f}")
