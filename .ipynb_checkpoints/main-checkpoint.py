import langchain
import tiktoken
import numpy as np
from langchain_openai import OpenAIEmbeddings
## Indexing
question = 'What is dog?'
document = 'Dog is an animal found in the world'
def num_tokens_from_string(question,encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_encoding= len(encoding.encode(question))
    return num_encoding
# print(num_tokens_from_string(question,'cl100k_base'))

embed = OpenAIEmbeddings()
query_result = embed.embed_query(question)
doc_result = embed.embed_query(document)
def cosine_similarity(vec1,vec2):
    dot_prdt = np.dot(vec1,vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_prdt/(norm1*norm2)
print(cosine_similarity(query_result,doc_result))