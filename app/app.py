from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from datetime import datetime
import re 
import streamlit as st 

# SETUP
PDF_STORAGE_PATH = 'docs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="mxbai-embed-large")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
PROMPT_TEMPLATE = """Você é um experiente assistente. Use o contexto fornecido para responder à pergunta.
Se você não tiver certeza, responda que não sabe.
Seja conciso e mantenha-se atento aos fatos (máximo de 3 sentenças).
Responda em português.

Pergunta: {user_query}
Contexto: {document_context}
Resposta:
"""


# Aux. Functions 
def extract_answer(text):
    final_answer = re.sub("<think>.*?</think>", " ", text, flags=re.DOTALL)
    return final_answer.strip()

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, 'wb') as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        add_start_index = True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query":user_query, "document_context":context_text})

# User Interface
st.set_page_config(page_title="DocMentor", page_icon="", layout='wide')

st.title("DocMentor")
st.markdown("Seu assistente de IA para documentos!")

st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Selecione um arquivo PDF", 
    type="pdf",
    help="Selecione um arquivo PDF para análise",
    accept_multiple_files=False
)

if uploaded_file:
    saved_path = save_uploaded_file(uploaded_file)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

    st.success("Documento carregado com sucesso")

    user_input = st.chat_input("Digite sua pergunta sobre o documento")

    if user_input:
        
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analisando documento..."):
            relevant_docs = find_related_documents(user_input)                                                                                          
            response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant"):                                                                                                             

            st.write(response)