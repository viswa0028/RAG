from operator import itemgetter
import os
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.load import loads, dumps
from langchain_community.document_loaders import PyPDFLoader

# Set Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = input("Enter your Google API key: ")

# Load PDF document
pdf_path = "./X Physics EM Title.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Create hierarchical chunks
# 1. First for sections (larger chunks)
section_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n\n", "\n\n", "\n", " ", ""]
)
sections = section_splitter.split_documents(documents)

# 2. Then for paragraphs (smaller chunks)
paragraph_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# Store both small and big chunks with their relationships
all_chunks = []
for section_idx, section in enumerate(sections):
    # Store the section (big chunk)
    section_id = f"section_{section_idx}"
    section_text = section.page_content

    # Try to extract a better title from the first line
    lines = section_text.strip().split("\n")
    section_title = f"Section {section_idx}"  # Default
    if lines and len(lines[0]) < 100:  # If first line is reasonably short, it might be a title
        section_title = lines[0]

    # Split section into paragraphs (small chunks)
    paragraphs = paragraph_splitter.create_documents(
        [section_text],
        metadatas=[section.metadata]
    )

    # Store each paragraph with reference to its parent section
    for para_idx, para_doc in enumerate(paragraphs):
        all_chunks.append({
            "text": para_doc.page_content,
            "is_small_chunk": "True",  # Store as string to avoid type issues
            "chunk_id": f"{section_id}_para_{para_idx}",
            "parent_id": section_id,
            "parent_title": section_title,
            "section_index": str(section_idx),  # Convert to string
            "paragraph_index": str(para_idx),  # Convert to string
            "page": str(section.metadata.get("page", "0"))  # Convert to string
        })

    # Also store the full section
    all_chunks.append({
        "text": section_text,
        "is_small_chunk": "False",  # Store as string
        "chunk_id": section_id,
        "parent_id": "",  # Empty string instead of None
        "section_title": section_title,
        "section_index": str(section_idx),  # Convert to string
        "page": str(section.metadata.get("page", "0"))  # Convert to string
    })

# Prepare data for vector store
texts = [chunk["text"] for chunk in all_chunks]


# Clean metadata to ensure no None values (ChromaDB requirement)
def clean_metadata(metadata_dict):
    cleaned = {}
    for k, v in metadata_dict.items():
        if k != "text":
            # Convert None to empty string or appropriate default value
            if v is None:
                v = ""
            # Ensure boolean, numeric, or string type
            if not isinstance(v, (bool, int, float, str)):
                v = str(v)
            cleaned[k] = v
    return cleaned


metadatas = [clean_metadata(chunk) for chunk in all_chunks]
ids = [chunk["chunk_id"] for chunk in all_chunks]

# Create embeddings
embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Store in ChromaDB
vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embed,
    metadatas=metadatas,
    ids=ids)


# Function to retrieve both small chunks and their parent sections
def small_to_big_retrieval(query, vector_store, k=3):
    # 1. First retrieve the most relevant small chunks
    retriever = vector_store.as_retriever(
        search_kwargs={
            'k': k,
            'filter': {"is_small_chunk": "True"}  # String value for boolean filter
        }
    )
    small_chunks = retriever.get_relevant_documents(query)

    # 2. Get the parent sections of these small chunks
    parent_ids = [doc.metadata.get("parent_id") for doc in small_chunks]

    # 3. Retrieve the full sections
    sections = []
    for parent_id in parent_ids:
        if parent_id:  # Make sure parent_id is not None or empty
            section_docs = vector_store.similarity_search(
                "",  # Empty query because we're filtering by ID
                k=1,
                filter={"chunk_id": parent_id}
            )
            if section_docs:
                sections.append(section_docs[0])

    # 4. Return both small chunks and their parent sections
    return {
        "small_chunks": small_chunks,
        "parent_sections": sections
    }


# Get user question
question = input("What is your question? ")

# Generate search variations using Gemini
prompt_template = """You are a helpful assistant that generates multiple search queries based on a single query input.

Generate multiple search queries related to: {question}

Output (4 queries):"""
prompt = ChatPromptTemplate.from_template(prompt_template)

generate_queries = (
        prompt
        | ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)


# RAG Fusion function
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


# Setup RAG fusion retrieval
retriever = vector_store.as_retriever(search_kwargs={'k': 3})
retrieval_rag_fusion = generate_queries | retriever.map() | rag_fusion

# Retrieve documents
retrieved_docs = retrieval_rag_fusion.invoke({"question": question})

# Final Answer generation
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7)

final_prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the context below:

Context:
{context}

Question: {question}

Provide a comprehensive and accurate answer based only on the information in the context.
"""
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

# Generate answer
answer = final_chain.invoke({"question": question})
print("\n Final Answer:\n", answer)

# Display retrieved context for debugging
print("\n📄 Retrieved Context Chunks:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Chunk {i + 1} ---\n{doc.page_content}\n")