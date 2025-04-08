import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# Cargar base de datos de productos
example_productos = [
    {"nombre": "Camisa Azul", "talla": "S", "modelo": "Slim Fit"},
    {"nombre": "Camisa Azul", "talla": "M", "modelo": "Regular Fit"},
    {"nombre": "Camisa Azul", "talla": "L", "modelo": "Oversized"},
    {"nombre": "Pantalón Negro", "talla": "M", "modelo": "Skinny"},
    {"nombre": "Pantalón Negro", "talla": "L", "modelo": "Regular"},
    {"nombre": "Zapatos Deportivos", "talla": "42", "modelo": "Nike Air"}
]

productos = [
    # --- Producto Maestro: Remera ---
    {"nombre": "Remera", "manga": "corta", "color": "azul", "cuello": "chomba", "gráfica": "lisa"},
    {"nombre": "Remera", "manga": "larga", "color": "rojo", "cuello": "cuello redondo", "gráfica": "estampadas"},
    {"nombre": "Remera", "manga": "corta", "color": "blanco", "cuello": "escote en V", "gráfica": "lisa"},
    
    # --- Producto Maestro: Buzo ---
    {"nombre": "Buzo", "estilo": "buzo", "color": "negro", "modelo": "canguro con capucha", "gráfica": "estampadas"},
    {"nombre": "Buzo", "estilo": "campera", "color": "amarillo", "modelo": "con capucha no canguro", "gráfica": "lisa"},
    
    # --- Producto: Cordón ---
    {"nombre": "Cordón", "color": "blanco", "longitud": "corto", "forma": "plano"},
    {"nombre": "Cordón", "color": "negro", "longitud": "largo", "forma": "redondo"},
    
    # --- Producto: Cierre ---
    {"nombre": "Cierre", "tipo": "Metálico", "largo": "12", "ancho": "2"},
    {"nombre": "Cierre", "tipo": "Sintético tejido", "largo": "10", "ancho": "1"},
    
    # --- Producto: Pantalón Jogging ---
    {"nombre": "Pantalón Jogging", "material": "Algodón", "corte": "Slim fit", "cintura": "Elástico"},
    {"nombre": "Pantalón Jogging", "material": "Dry Fit", "corte": "Oversized", "cintura": "Cordón ajustable"},
    
    # --- Producto: Elástico ---
    {"nombre": "Elástico", "material": "Elástico de poliéster", "ancho": "Elástico medio", "grosor": "Firme", "tipo": "Tejido", "diseño": "Reflectivo"},
    {"nombre": "Elástico", "material": "Elástico de látex", "ancho": "Elástico fino", "grosor": "Suave", "tipo": "Liso", "diseño": "Negro"},
    
    # --- Producto: Media Corta ---
    {"nombre": "Media corta", "material": "Algodón", "tipo": "Tobillera", "grosor": "Intermedias", "estilo": "Lisos"},
    {"nombre": "Media corta", "material": "Lana", "tipo": "Media deportiva", "grosor": "Gruesas o térmicas", "estilo": "Rayados o con patrones geométricos"}
]


# Convertir productos a texto para embeddings
#documentos = [f"{p['nombre']} - Talla: {p['talla']}, Modelo: {p['modelo']}" for p in productos]
documentos = [
    f"{p['nombre']} - " + ", ".join([f"{k.capitalize()}: {v}" for k, v in p.items() if k != 'nombre'])
    for p in productos
]

# Crear embeddings y base de datos FAISS
#embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
vectorstore = FAISS.from_texts(documentos, embeddings)

# Interfaz con Streamlit
st.title("Recomendador de Productos")
query = st.text_input("¿Qué producto buscas?")

if query:
    docs = vectorstore.similarity_search(query, k=5)  # Buscar productos similares
    st.subheader("Resultados Recomendados:")
    for doc in docs:
        st.write(doc.page_content)
