import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

from io import BytesIO


st.set_page_config(page_title="Chat With One or More PDFs")

load_dotenv()

genai.configure(api_key=os.getenv("Google_API_KEY"))


from io import BytesIO

def getPdfText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# def getPdfText(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf.seek(0)  # Ensure the file pointer is at the start
#         pdf_reader = PdfReader(BytesIO(pdf.read()))  # Wrap the bytes in a BytesIO object
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def getPdfText(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader=PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# dividing the texts into chuncks
def getTextChunks(text):
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
    # return vector_store

def get_conversational_chain():
    Prompt_Template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "You're Stupid & answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
""" 
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template=Prompt_Template, input_variables = ["context","question"])
    chain = load_qa_chain(model, chain_type = "stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "model/embedding-001")

    new_db = FAISS.load_local("faiss_index",embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents":docs, "question": user_question
        }, return_only_outputs = True)
    print(response)
    st.write("Reply: ", response["output_text"])
    # st.write("Context: ", response["context"])


def main(): #streamlit app UI part
    st.title("Chat with Pdf")
    # st.set_page_config(page_title="Chat With One or More PDFs")
    # st.set_page_config(page_title="Chat With One or More PDFs")

    st.header("Chat with PDF using Gemini Pro")

    user_question = st.text_input("Ask question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("upload your PDF files and Submit")
        if st.button("SUbmit & process"):
            with st.spinner("Processing..."):
                raw_text = getPdfText(pdf_docs)
                text_chunks = getTextChunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()

