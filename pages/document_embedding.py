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
st.set_page_config(page_title="Document Embedding", page_icon="📚", layout="wide")

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

st.title("📚 Document Embedding")
st.markdown("This page is used to upload the documents as the custom knowledge for the chatbot.")

with st.form("document_input"):
    st.markdown("### Upload Document")
    document = st.file_uploader("Knowledge Documents", type=['pdf', 'txt'], help=".pdf or .txt file")

    st.markdown("### Embedding Settings")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        instruct_embeddings = st.text_input(
            "Model Name of the Instruct Embeddings", value="hkunlp/instructor-xl"
        )
    
    with col2:
        chunk_size = st.number_input(
            "Chunk Size", value=200, min_value=0, step=1,
        )
    
    with col3:
        chunk_overlap = st.number_input(
            "Chunk Overlap", value=10, min_value=0, step=1, help="higher than chunk size"
        )
    
    st.markdown("### Vector Store Settings")
    col4, col5 = st.columns(2)
    with col4:
        vector_store_list = os.listdir("/content/drive/MyDrive/PDFtota/vector store/")
        vector_store_list = ["<New>"] + vector_store_list
        
        existing_vector_store = st.selectbox(
            "Vector Store to Merge the Knowledge", vector_store_list,
            help="Which vector store to add the new documents. Choose <New> to create a new vector store."
        )

    with col5:
        new_vs_name = st.text_input(
            "New Vector Store Name", value="new_vector_store_name",
            help="If choose <New> in the dropdown / multiselect box, name the new vector store. Otherwise, fill in the existing vector store to merge."
        )

    save_button = st.form_submit_button("Save vector store")

if save_button:
    with st.spinner("Processing document..."):
        if document.name[-4:] == ".pdf":
            document = read_pdf(document)
        elif document.name[-4:] == ".txt":
            document = read_txt(document)
        else:
            st.error("Check if the uploaded file is .pdf or .txt")

        split = split_doc(document, chunk_size, chunk_overlap)

        create_new_vs = None
        if existing_vector_store == "<New>" and new_vs_name != "":
            create_new_vs = True
        elif existing_vector_store != "<New>" and new_vs_name != "":
            create_new_vs = False
        else:
            st.error("Check the 'Vector Store to Merge the Knowledge' and 'New Vector Store Name'")
        
        embedding_storing(
            instruct_embeddings, split, create_new_vs, existing_vector_store, new_vs_name
        )

st.markdown("---")
st.markdown("### Existing Vector Stores")
st.json(os.listdir("/content/drive/MyDrive/PDFtota/vector store/"))
