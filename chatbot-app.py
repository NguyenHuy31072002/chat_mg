import logging
from collections import defaultdict
import tempfile
from time import time
from os import getenv, getcwd, path
from os.path import getsize
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv 
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.cache import MongoDBCache

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo import monitoring

# Logging setup
log_file_path = path.join(getcwd(), "chatbot-app-logs.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_file_path)]
)

# Streamlit page configuration set up
st.set_page_config(page_title="Chat with PDF Manuals", page_icon=":telephone_receiver:")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load environment variables
load_dotenv()

# MongoDB Event Listeners
class AggregatedCommandLogger(monitoring.CommandListener):
    def __init__(self):
        self.operation_counts = defaultdict(int)
        self.total_duration = 0
        self.total_operations = 0

    def started(self, event):
        pass  # not logging command start events

    def succeeded(self, event):
        self.operation_counts[event.database_name] += 1
        self.total_duration += event.duration_micros
        self.total_operations += 1

    def failed(self, event):
        database_name = event.__dict__.get('database_name', 'unknown')
        logging.info(f"Command failed: operation_id={event.operation_id}, duration_micros={event.duration_micros}, database_name={database_name}")

    def summarize_and_reset(self):
        if self.total_operations > 0:
            avg_duration = self.total_duration / self.total_operations
            summary = f"MongoDB operations summary: {self.total_operations} total operations, "
            summary += f"average duration: {avg_duration:.2f} microseconds. "
            summary += "Operations per database: " + ", ".join(f"{db}: {count}" for db, count in self.operation_counts.items())
            logging.info(summary)

        # Reset counters
        self.operation_counts.clear()
        self.total_duration = 0
        self.total_operations = 0

# Global logger instance
aggregated_logger = AggregatedCommandLogger()

# Register the command logger with PyMongo's monitoring framework
monitoring.register(aggregated_logger)

# Function to call at the end of your script
def log_mongodb_summary():
    aggregated_logger.summarize_and_reset()

# Environment Variables and MongoDB Connection
ATLAS_URI = getenv("ATLAS_URI")
MONGODB_DB = getenv("MONGODB_DB")
MONGODB_COLLECTION = getenv("MONGODB_COLLECTION")


client = MongoClient(ATLAS_URI, server_api=ServerApi('1'))

def test_mongodb_connection():
    """A function to test the MongoDB Connection inside the application"""
    try:
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")


# Set up MongoDB cache
set_llm_cache(MongoDBCache(
    connection_string=ATLAS_URI,
    collection_name="llm_cache",  
    database_name=MONGODB_DB
))

# Function for single page OCR processing
def ocr_single_page(image):
    return pytesseract.image_to_string(image)

def ocr_on_pdf(pdf_path):
    try:
        pytesseract.pytesseract.tesseract_cmd = getenv("TESSERACT_PATH")
        images = convert_from_path(pdf_path)
        file_size = path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        
        if file_size > 5:  # If file is larger than 5MB
            with ThreadPoolExecutor() as executor:
                extracted_texts = list(executor.map(ocr_single_page, images))
            extracted_text = "\n".join(extracted_texts)
            logging.info(f"Parallel OCR completed for large file: {pdf_path}")
        else:
            extracted_text = "\n".join(ocr_single_page(image) for image in images)
            logging.info(f"Sequential OCR completed for small file: {pdf_path}")
        
        return extracted_text
    except Exception as e:
        logging.error(f"Error during OCR on PDF: {e}")
        return ""

def process_pdf(pdf):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(pdf.read())
            temp_pdf_path = temp_pdf.name

        file_size = getsize(temp_pdf_path) / (1024 * 1024)  # Size in MB
        logging.info(f"Processing PDF: {pdf.name}, Size: {file_size:.2f} MB")

        if file_size == 0:
            logging.warning(f"The PDF file '{pdf.name}' is empty.")
            return ""

        pdf_reader = PdfReader(temp_pdf_path)

        try:
            text_from_pdf = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        except Exception as e:
            # Catch specific exception for AES encryption
            if "cryptography>=3.1 is required for AES algorithm" in str(e):
                logging.warning(f"PDF '{pdf.name}' is AES encrypted. Performing OCR.")
                return ocr_on_pdf(temp_pdf_path)
            else:
                raise e

        if not text_from_pdf:
            logging.warning(f"No text extracted from '{pdf.name}'. Performing OCR.")
            return ocr_on_pdf(temp_pdf_path)

        logging.info(f"Processed PDF: {pdf.name}")
        return text_from_pdf

    except Exception as e:
        logging.error(f"Error processing PDF: {pdf.name}. Error: {e}")
        return ""



def get_pdf_text(pdf_docs):
    return "".join(process_pdf(pdf) for pdf in pdf_docs)




def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = [chunk for chunk in text_splitter.split_text(text)]
    logging.info(f"Amount of chunks created: {len(chunks)}")
    return chunks


from langchain_community.embeddings import SentenceTransformerEmbeddings


def get_vectorstore(text_chunks: List[str], metadatas: List[Dict[str, Any]] = None) -> MongoDBAtlasVectorSearch:
    embeddings = SentenceTransformerEmbeddings(model_name='keepitreal/vietnamese-sbert')

    mongo_client = MongoClient(ATLAS_URI)
    db = mongo_client[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]

    vector_search = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine"
    )

    try:
        ids = [vector_search.add_texts([chunk], [metadata] if metadatas else None)[0] 
               for chunk, metadata in zip(text_chunks, metadatas or [None] * len(text_chunks))]
        logging.info(f"Added {len(ids)} embeddings to the vector store: {ids}")
    except Exception as e:
        logging.error(f"Error adding embeddings: {str(e)}")

    return vector_search





# Tạo chuỗi hỏi đáp
def create_qa_chain(db):
    def custom_llm(query, context):
        api_key = getenv("GOOGLE_API_KEY")
        # Load API key từ .env
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel('gemini-1.5-flash')
        # Tạo câu hỏi từ ngữ cảnh và câu hỏi
        full_prompt = f"Câu hỏi: {query}\nNgữ cảnh: {context}\nTrả lời:"
        response = llm.generate_content(full_prompt)
        if "Câu trả lời không có trong ngữ cảnh" in response.text:
            response = llm.generate_content(query)
        return response.text

    class CustomRetrievalQA:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, inputs):
            query = inputs["question"]
            docs = self.retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])
            return {"answer": custom_llm(query, context)}

    retriever = db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=6000)
    return CustomRetrievalQA(retriever)

import google.generativeai as genai

def get_conversation_chain(vectorstore):
    # Lấy API key từ biến môi trường và cấu hình Google Gemini API
    api_key = getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash')
    
    # Tạo đối tượng CustomRetrievalQA
    custom_qa_chain = create_qa_chain(vectorstore)
    
    return custom_qa_chain


# def get_conversation_chain(vectorstore):
#     # llm = ChatOpenAI(model_name="gpt-4")
#     # Lấy API key từ biến môi trường và cấu hình Google Gemini API
#     api_key = getenv("GOOGLE_API_KEY")
#     # Load API key từ .env
#     genai.configure(api_key=api_key)
#     llm = genai.GenerativeModel('gemini-1.5-flash')
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs first or wait until the database is initialized.")
    else:
        with get_openai_callback() as cb:
            response = st.session_state.conversation.invoke({"question": user_question})
            st.session_state.chat_history.append({"type": "user", "content": user_question})
            st.session_state.chat_history.append({"type": "bot", "content": response["answer"]})
            logging.info(f"\n\tOpenAI Token Usage:\n\t{cb}")

def clear_chat_history():
    logging.info("Clearing chat history")
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.rerun()

def main():
    prog_start_time = time()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if st.session_state.vectorstore is None:
        db = client[MONGODB_DB]  # Use the global client
        collection = db[MONGODB_COLLECTION]
        # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        embeddings = SentenceTransformerEmbeddings(model_name='keepitreal/vietnamese-sbert')
        st.session_state.vectorstore = MongoDBAtlasVectorSearch(
            collection,
            embedding=embeddings,  
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
            relevance_score_fn="cosine"
        )

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    # Add a sidebar for the PDF upload box
    with st.sidebar:
        st.subheader("Your PDF Manuals")
        uploaded_files = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf"])
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Processing complete.")

    # Main page content
    st.title(":rainbow[Chat with PDF Manuals]  :open_book:")
    user_question = st.chat_input("Ask a question about your PDFs:")
    if user_question:
        handle_userinput(user_question)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user" if message["type"] == "user" else "assistant"):
            st.write(message["content"])

    # Clear chat history button
    if st.button("Clear Chat History"):
        clear_chat_history()

    exec_time = time() - prog_start_time
    logging.info(f"Script executed in {exec_time:.2f} seconds.")
    log_mongodb_summary()


if __name__ == '__main__':
    test_mongodb_connection()
    main()