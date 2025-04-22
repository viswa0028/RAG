import langchain
import tiktoken
import numpy as np
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
