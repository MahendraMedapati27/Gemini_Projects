import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import GooglePalmEmbeddings
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_response(file, query):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in file)
    data = text_splitter.split_text(context)
    embeddings = GooglePalmEmbeddings(model="models/embedding-001")
    searcher = Chroma.from_texts(data, embeddings).as_retriever()
    records = searcher.invoke(query)
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain.invoke({
        "input_documents": records,
        "question": query
    })
    
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat with CSV using Gemini Pro")
    st.title("Chat with CSV using Gemini Pro")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        with st.spinner("Processing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                csv_loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                data = csv_loader.load()
                user_input = st.text_input("Your question: ")
                
                if user_input:
                    response = get_response(data, user_input)
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.remove(tmp_file_path)
    else:
        st.info("Please upload a CSV file.")
            
if __name__ == "__main__":
    main()
