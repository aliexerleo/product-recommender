import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI

from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()


# --- Lista de productos
productos = [
    {"nombre": "Remera", "manga": "corta", "color": "azul", "cuello": "chomba", "gráfica": "lisa"},
    {"nombre": "Remera", "manga": "larga", "color": "rojo", "cuello": "cuello redondo", "gráfica": "estampadas"},
    {"nombre": "Remera", "manga": "corta", "color": "blanco", "cuello": "escote en V", "gráfica": "lisa"},
    {"nombre": "Buzo", "estilo": "buzo", "color": "negro", "modelo": "canguro con capucha", "gráfica": "estampadas"},
    {"nombre": "Buzo", "estilo": "campera", "color": "amarillo", "modelo": "con capucha no canguro", "gráfica": "lisa"},
    {"nombre": "Cordón", "color": "blanco", "longitud": "corto", "forma": "plano"},
    {"nombre": "Cordón", "color": "negro", "longitud": "largo", "forma": "redondo"},
    {"nombre": "Cierre", "tipo": "Metálico", "largo": "12", "ancho": "2"},
    {"nombre": "Cierre", "tipo": "Sintético tejido", "largo": "10", "ancho": "1"},
    {"nombre": "Pantalón Jogging", "material": "Algodón", "corte": "Slim fit", "cintura": "Elástico"},
    {"nombre": "Pantalón Jogging", "material": "Dry Fit", "corte": "Oversized", "cintura": "Cordón ajustable"},
    {"nombre": "Elástico", "material": "Elástico de poliéster", "ancho": "Elástico medio", "grosor": "Firme", "tipo": "Tejido", "diseño": "Reflectivo"},
    {"nombre": "Elástico", "material": "Elástico de látex", "ancho": "Elástico fino", "grosor": "Suave", "tipo": "Liso", "diseño": "Negro"},
    {"nombre": "Media corta", "material": "Algodón", "tipo": "Tobillera", "grosor": "Intermedias", "estilo": "Lisos"},
    {"nombre": "Media corta", "material": "Lana", "tipo": "Media deportiva", "grosor": "Gruesas o térmicas", "estilo": "Rayados o con patrones geométricos"}
]

# --- Crear documentos para embeddings
documentos = [
    f"{p['nombre']} - " + ", ".join([f"{k.capitalize()}: {v}" for k, v in p.items() if k != 'nombre'])
    for p in productos
]

# --- Crear embeddings y base de datos
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
vectorstore = FAISS.from_texts(documentos, embeddings)

# --- Crear QA chain (usando un modelo LLM, como OpenAI)
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", model="gpt-4o-mini", temperature=1.0),
    chain_type="stuff",
    retriever=retriever
)


# --- Interfaz de chat con Streamlit
st.set_page_config(page_title="Chat Q&A de Productos", layout="wide")
st.title("🛍️ Chat de Preguntas sobre Productos")

# Inicializar historial si no existe
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar historial
for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
    st.chat_message("user", avatar="🧑").write(user_msg)
    st.chat_message("assistant", avatar="🤖").write(bot_msg)

# Input del usuario
user_input = st.chat_input("Preguntá lo que quieras sobre los productos...")

if user_input:
    # Mostrar mensaje del usuario
    st.chat_message("user", avatar="🧑").write(user_input)

    # Generar respuesta
    respuesta = qa.run(user_input)

    # Mostrar respuesta
    st.chat_message("assistant", avatar="🤖").write(respuesta)

    # Guardar en historial
    st.session_state.chat_history.append((user_input, respuesta))