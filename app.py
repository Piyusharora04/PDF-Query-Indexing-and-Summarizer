from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pdfplumber
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor


from langchain_community.document_loaders import PDFMinerLoader

from tempfile import NamedTemporaryFile


checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    device_map='cpu', 
    torch_dtype=torch.float32,
    offload_folder="offload"
)

@st.cache_resource
def load_embeddings_model():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    embeddings = load_embeddings_model()
    llm = llm_pipeline()
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

@st.cache_resource
def summarizer_model():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

def summarize_text_from_pdf(file_stream):
    summarizer = summarizer_model()
    text = ""
    
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    chunk_size = 512  
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def summarize_chunk(chunk):
        summary = summarizer(
            chunk,
            max_length=150,  
            min_length=50,
            do_sample=False
        )
        return summary[0]['summary_text']
    
    with ThreadPoolExecutor() as executor:
        summary_list = list(executor.map(summarize_chunk, text_chunks))
    
    full_summary = ' '.join(summary_list)
    
    return full_summary


############################################################################

def pdf_querying():
    st.title("Search Your PDF 🐦📄")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            """
        )
        
    text = ""
    
    uploaded_files = st.file_uploader("Choose files ", type='pdf', accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # file_name = uploaded_file # This will use the original file name
            # upload_file_to_firebase(file_name=file_name)
            # document = download_and_process_pdf(uploaded_file)
            # documents.extend(document)
            with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
                temp.write(uploaded_file.getvalue())
                temp.flush()
                temp.seek(0)
                # print(temp.name)
                loader = PDFMinerLoader(temp.name)
                documents.extend(loader.load())
                
        
        if documents:
            embeddings = load_embeddings_model()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)

            # Embeddings...
            db = Chroma.from_documents(
                texts, embeddings, persist_directory="db"
            )
            print("Ingestion completed successfully!")
        else:
            print("No documents were loaded for processing.")
    
    
    
    question = st.text_area("Enter your Question")
    if st.button("Ask"):
        st.info("Your Question: " + question)
        st.info("Your Answer")
        answer, extra = process_answer(question)
        st.write(answer)
        # st.write(extra)

########################################################################

def pdf_summary():
    st.title("Summarize Your PDF 🐦📄")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered app that can summarize your large PDF File.
            """
        )
    

    uploaded_file = st.file_uploader("Choose files ", type='pdf', accept_multiple_files=False)
    if uploaded_file:
        summary = ""
        with NamedTemporaryFile(suffix="pdf", delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            temp.flush()
            temp.seek(0)
            if st.button(f"Summarize PDF"):
                summary = summarize_text_from_pdf(temp)
                st.write(summary)            


def main():
    st.sidebar.title("Welcome to the ultimate PDF Playing app😊")
    
    add_selectbox = st.sidebar.selectbox(
        "What would you like to do...?",
        ("PDF Querying", "PDF Summarization")
    )
    
    if add_selectbox == "PDF Querying":
        pdf_querying()
    elif add_selectbox == "PDF Summarization":
        pdf_summary()

if __name__ == '__main__':
    main()
