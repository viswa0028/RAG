from operator import itemgetter
import os
import numpy as np
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.load import loads, dumps
from langchain_community.document_loaders import PyPDFLoader

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = input("Enter your Google API key: ")

loader = PyPDFLoader("./X Physics EM Title.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma.from_documents(split_docs, embedding=embed)

#Retrieve based on question
question = 'What is reflection?'
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

#Generate search variations using Gemini
prompt_template = """You are a helpful assistant that generates multiple search queries based on a single query input.

Generate multiple search queries related to: {question}

Output (4 queries):"""
prompt = ChatPromptTemplate.from_template(prompt_template)

generate_queries = (
    prompt
    | ChatGoogleGenerativeAI(model = 'gemini-2.0-flash',temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

#RAG Fusion
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

#retrieval
retrieved_docs = retrieval_rag_fusion.invoke({"question": question})

# Final Answer
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
print("\n Final Answer:\n", answer)

# Retrieved raw data
print("\n📄 Retrieved Context Chunks:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Chunk {i+1} ---\n{doc.page_content}\n")

