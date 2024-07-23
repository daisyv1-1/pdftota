import os
import streamlit as st
from langchain.document_loaders import TextLoader
from pypdf import PdfReader
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Apply custom styles
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF; /* Set the main background to white */
        color: #000000; /* Ensure text color is black for readability */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stSelectbox>div>div>select {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stNumberInput>div>div>input {
        background-color: #FFFFFF;
        color: #000000;
    }
    .css-1r6slb1 { /* Sidebar background */
        background-color: #2E2E2E; /* Dark background for the sidebar */
        color: #FFFFFF; /* White text in the sidebar */
    }
    .css-1r6slb1 .css-1g5s1x6 { /* Sidebar text color */
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Rest of the functions remain unchanged
def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document

def split_doc(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split

def embedding_storing(model_name, split, create_new_vs, existing_vector_store, new_vs_name):
    if create_new_vs is not None:
        instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=model_name)
        db = FAISS.from_documents(split, instructor_embeddings)
        if create_new_vs:
            db.save_local("/content/drive/MyDrive/PDFtota/vector store/" + new_vs_name)
        else:
            load_db = FAISS.load_local(
                "/content/drive/MyDrive/PDFtota/vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            load_db.merge_from(db)
            load_db.save_local("/content/drive/MyDrive/PDFtota/vector store/" + new_vs_name)
        st.success("The document has been saved.")

def prepare_rag_llm(token, llm_model, instruct_embeddings, vector_store_list, temperature, max_length):
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=instruct_embeddings, model_kwargs={"device":"cuda"}
    )
    loaded_db = FAISS.load_local(
        f"/content/drive/MyDrive/PDFtota/vector store/{vector_store_list}", 
        instructor_embeddings, 
        allow_dangerous_deserialization=True
    )
    llm = HuggingFaceHub(
        repo_id=llm_model,
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=token
    )
    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(),
        return_source_documents=True,
        memory=memory,
    )
    return qa_conversation

def generate_answer(question, token):
    if token == "":
        answer = "Insert the Hugging Face token"
        doc_source = ["no source"]
    else:
        response = st.session_state.conversation({"question": question})
        answer = response.get("answer").split("Helpful Answer:")[-1].strip()
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]
    return answer, doc_source

st.title("🤖 Talk to your PDF")

with st.expander("Setting the LLM", expanded=True):
    st.markdown("This page is used to have a chat with the uploaded documents")
    with st.form("setting"):
        col1, col2, col3 = st.columns(3)
        with col1:
            token = st.text_input("Hugging Face Token", type="password")
        with col2:
            llm_model = st.text_input("LLM model", value="tiiuae/falcon-7b-instruct")
        with col3:
            instruct_embeddings = st.text_input("Instruct Embeddings", value="hkunlp/instructor-xl")
        
        col4, col5, col6 = st.columns(3)
        with col4:
            vector_store_list = os.listdir("/content/drive/MyDrive/PDFtota/vector store/")
            default_choice = (
                vector_store_list.index('naruto_snake')
                if 'naruto_snake' in vector_store_list
                else 0
            )
            existing_vector_store = st.selectbox("Vector Store", vector_store_list, default_choice)
        with col5:
            temperature = st.number_input("Temperature", value=1.0, step=0.1)
        with col6:
            max_length = st.number_input("Maximum character length", value=300, step=1)
        
        create_chatbot = st.form_submit_button("Create chatbot")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if token:
    st.session_state.conversation = prepare_rag_llm(
        token, llm_model, instruct_embeddings, existing_vector_store, temperature, max_length
    )

if "history" not in st.session_state:
    st.session_state.history = []

if "source" not in st.session_state:
    st.session_state.source = []

st.markdown("### Chat History")
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    answer, doc_source = generate_answer(question, token)
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

with st.expander("Source documents"):
    st.json(st.session_state.source)
