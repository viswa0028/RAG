from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.load import loads, dumps
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from utils import retrieve_from_kg
import os

def query_rag(question, persist_directory="./chroma_db"):
    # Ensure Google API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = input("Enter your Google API key: ")

    # Initialize vector store
    embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ["GOOGLE_API_KEY"])
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embed)

    # Generate search variations
    prompt_template = """You are a helpful assistant that generates multiple search queries based on a single query input.

Generate multiple search queries related to: {question}

Output (4 queries):"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    generate_queries = (
        prompt
        | ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0, google_api_key=os.environ["GOOGLE_API_KEY"])
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

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"])

    # Retrieve from Knowledge Graph
    print("🔍 Querying Knowledge Graph...")
    kg_info = retrieve_from_kg(question, llm=llm)

    # Combine contexts
    vector_context = "\n".join(doc.page_content for doc in retrieved_docs)
    kg_context = kg_info.get("answer", "")

    # Generate final answer
    final_prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context below from both vector database and knowledge graph:

Vector Database Context:
{vector_context}

Knowledge Graph Context:
{kg_context}

Question: {question}

Provide a comprehensive and accurate answer based on the information from both contexts. 
If the knowledge graph provides structured relationships or entities relevant to the question, 
incorporate that information along with the detailed content from the vector database.
"""
    )

    final_chain = (
        {
            "vector_context": lambda x: vector_context,
            "kg_context": lambda x: kg_context,
            "question": itemgetter("question")
        }
        | final_prompt
        | llm
        | StrOutputParser()
    )

    # Generate answer
    answer = final_chain.invoke({"question": question})
    print("\n🎯 Final Answer:\n", answer)

    # Display context for debugging
    print("\n📄 Retrieved Vector Context Chunks:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {i + 1} ---\n{doc.page_content}\n")

    if kg_context:
        print("\n🕸️ Knowledge Graph Context:")
        print(kg_context)

    if kg_info.get("cypher_query"):
        print(f"\n🔍 Generated Cypher Query:")
        print(kg_info["cypher_query"])

if __name__ == "__main__":
    question = input("What is your question? ")
    query_rag(question)