import langchain
import tiktoken
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# from PyQt5.QtWebEngineWidgets import kwargs
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from sympy.physics.units import temperature

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = input("Enter your Google API key: ")

question = 'What is dog?'
document = 'Dog is an animal found in the world'

# Token count helper (optional)
def num_tokens_from_string(text, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

# Embeddings
embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
query_result = embed.embed_query(question)
doc_result = embed.embed_query(document)

# Cosine similarity
def cosine_similarity(vec1, vec2):
    dot_prdt = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_prdt / (norm1 * norm2)

print("Cosine Similarity:", cosine_similarity(query_result, doc_result))

#loading docs
# #load docs
#splitting text
text_splitting = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 300,
    chunk_overlap = 50
)

vector_store = Chroma.from_documents(document ,embedding=embed) #document should be the documents that are loaded

##Retreival

retriever = vector_store.as_retriever(search_kwargs ={'k':3})
docs= retriever.get_relevant_documents(question)
## docs will produce the output for our question

# ###Generation

template = """Answer the question based on the context:{context}
Question :{question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm= ChatGoogleGenerativeAI(model = 'gemini-pro',temperature=0.7)
chain = llm | prompt
chain.invoke({"context": document, "question": question})

