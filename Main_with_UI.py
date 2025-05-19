import os
import streamlit as st
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.load import loads, dumps
from langchain_community.document_loaders import PyPDFLoader
import tempfile

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")
st.markdown("Ask any question based on the uploaded PDF.")

api_key = st.text_input("Enter API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    with st.spinner(" Understanding"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)

        embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = Chroma.from_documents(split_docs, embedding=embed)

        retriever = vector_store.as_retriever(search_kwargs={'k': 3})

        query_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple search queries based on a single query input.
            Generate multiple search queries related to: {question}
            Output (4 queries):"""
        )
        generate_queries = (
            query_prompt
            | ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        def rag_fusion(results: list[list], k=60):
            fused_score = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_score:
                        fused_score[doc_str] = 0
                    fused_score[doc_str] += 1 / (rank + k)
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_score.items(), key=lambda x: x[1], reverse=True)
            ]
            return [doc for doc, _ in reranked_results]

        retrieval_rag_fusion = generate_queries | retriever.map() | rag_fusion

        question = st.text_input("Enter your question")

        if question:
            with st.spinner("Retrieving and Generating Answers"):
                retrieved_docs = retrieval_rag_fusion.invoke({"question": question})

                llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7)

                final_prompt = ChatPromptTemplate.from_template(
                    "Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
                )

                final_chain = (
                    {
                        "context": lambda x: "\n".join(doc.page_content for doc in retrieved_docs),
                        "question": itemgetter("question")
                    }
                    | final_prompt
                    | llm
                    | StrOutputParser()
                )

                answer = final_chain.invoke({"question": question})
                st.success("Answer Generated")
                st.markdown(f"### 🧠 Answer:\n{answer}")

                with st.expander("📄 Retrieved Context Chunks"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}:**\n```\n{doc.page_content}\n```")
else:
    st.warning("👆 Please upload a PDF and enter your API key to continue.")
