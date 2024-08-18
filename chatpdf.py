import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from io import BytesIO

st.set_page_config(page_title="Chat With One or More PDFs")

load_dotenv()

genai.configure(api_key=os.getenv("Google_API_KEY"))

def getPdfText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                st.write(f"Extracted text: {page_text[:500]}")  # Show the first 500 characters for inspection
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text


# def getPdfText(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(BytesIO(pdf.read()))  # Read the content as bytes and wrap it in BytesIO
#             for page in pdf_reader.pages:
#                 text += page.extract_text() or ""
#         except Exception as e:
#             st.error(f"Error reading PDF: {e}")
#     return text

def getTextChunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding")
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-1.5-flash-latest")
    # embeddings = GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash-latest")/
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    Prompt_Template = """ 
Answer the question as detailed as possible based on the provided context. If the context does not contain the information, simply respond with "You're Stupid & Answer not available in the context". Do not fabricate an answer.\n\n
Context:\n {context}\n
Question: \n{question}\n
Answer:
"""

#     Prompt_Template = """ 
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "You're Stupid! answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
# """ 
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {
                "input_documents": docs, "question": user_question
            }, return_only_outputs=True)
        # st.write("Response:", response)  # Print the entire response object
        st.write("Reply: ", response.get("output_text", "No output text found"))
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# def user_input(user_question):
#     # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-1.5-flash-latest")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     try:
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question)
#         chain = get_conversational_chain()
#         response = chain(
#             {
#                 "input_documents": docs, "question": user_question
#             }, return_only_outputs=True)
#         st.write("Reply: ", response["output_text"])
#     except ValueError as e:
#         st.error(f"Error loading FAISS index: {e}")
#     except GoogleGenerativeAIError as e:
#         st.error(f"Error with Google Generative AI: {e}")
#     except Exception as e:
#         st.error(f"Unexpected error: {e}")

def main():
    st.title("Chat with Pdf")
    st.header("Chat with PDF using Gemini Pro!!")
    user_question = st.text_input("Ask question from the PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files and Submit", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    try:
                        raw_text = getPdfText(pdf_docs)
                        text_chunks = getTextChunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done!")
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                else:
                    st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()