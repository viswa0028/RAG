from operator import itemgetter
import os
import json
import re
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.load import loads, dumps
from langchain_community.document_loaders import PyPDFLoader
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

# Set Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = input("Enter your Google API key: ")

# Neo4j connection setup
NEO4J_URI = "YOUR BOLT URL"
NEO4J_USERNAME = "YOUR USERNAME"
NEO4J_PASSWORD = "YOUR PASSWORD"

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)


# Free Knowledge Graph Generator using Gemini
class FreeKGGenerator:
    def __init__(self, api_key=None, model='gemini-2.0-flash', temperature=0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key or os.environ.get("GOOGLE_API_KEY")
        )

        # Prompt template for knowledge graph extraction
        self.kg_prompt = ChatPromptTemplate.from_template("""
You are an expert knowledge graph extractor. Extract entities and relationships from the given text.

Text: {text}

Extract information in the following JSON format:
{{
    "entities": [
        {{"name": "entity_name", "type": "entity_type", "description": "brief_description"}},
        ...
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "type": "relationship_type", "description": "brief_description"}},
        ...
    ]
}}

Focus on:
- Key concepts, theories, formulas, laws, principles
- Scientists, researchers, inventors
- Physical quantities, units, measurements
- Phenomena, processes, experiments
- Mathematical relationships and equations

Return only valid JSON without any additional text or formatting.
""")

    def generate(self, text):
        """Generate knowledge graph from text"""
        try:
            # Create the chain
            chain = self.kg_prompt | self.llm | StrOutputParser()

            # Get response
            response = chain.invoke({"text": text})

            # Clean and parse JSON
            response = response.strip()

            # Remove any markdown formatting
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            # Parse JSON
            try:
                kg_data = json.loads(response)
                return kg_data
            except json.JSONDecodeError:
                # Try to extract JSON from response using regex
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    kg_data = json.loads(json_match.group())
                    return kg_data
                else:
                    print(f"Could not parse JSON from response: {response[:200]}...")
                    return {"entities": [], "relationships": []}

        except Exception as e:
            print(f"Error in knowledge graph generation: {e}")
            return {"entities": [], "relationships": []}


# Initialize the free KG generator
kg = FreeKGGenerator()

# Load PDF document
pdf_path = "./initial documentation.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Create hierarchical chunks
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
kg_texts = []  # Store texts for knowledge graph generation

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

    # Add section text for knowledge graph generation
    kg_texts.append(section_text)

# Generate Knowledge Graph and store in Neo4j
print("🔄 Generating Knowledge Graph using Free Alternative...")
try:
    # Clear existing data (optional - remove if you want to append)
    graph.query("MATCH (n) DETACH DELETE n")

    # Generate knowledge graph from each section (process in smaller batches)
    batch_size = 3  # Smaller batch size for free API limits
    total_entities = 0
    total_relationships = 0

    for batch_start in range(0, len(kg_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(kg_texts))
        batch_texts = kg_texts[batch_start:batch_end]

        print(
            f"Processing batch {batch_start // batch_size + 1}/{(len(kg_texts) - 1) // batch_size + 1} (sections {batch_start + 1}-{batch_end})...")

        for i, text in enumerate(batch_texts):
            section_idx = batch_start + i
            try:
                # Limit text length to avoid API limits
                if len(text) > 3000:
                    text = text[:3000] + "..."

                # Generate knowledge graph
                kg_result = kg.generate(text)

                if kg_result and isinstance(kg_result, dict):
                    entities = kg_result.get('entities', [])
                    relationships = kg_result.get('relationships', [])

                    # Create entities in Neo4j
                    for entity in entities:
                        try:
                            entity_name = str(entity.get('name', '')).replace("'", "''").replace('"', '""')
                            entity_type = str(entity.get('type', 'Entity')).replace("'", "''").replace('"', '""')
                            entity_description = str(entity.get('description', '')).replace("'", "''").replace('"',
                                                                                                               '""')

                            if entity_name and len(entity_name.strip()) > 0:
                                query = f"""
                                MERGE (e:Entity {{name: '{entity_name}'}})
                                SET e.type = '{entity_type}',
                                    e.description = '{entity_description}',
                                    e.source_section = {section_idx}
                                """
                                graph.query(query)
                                total_entities += 1
                        except Exception as entity_error:
                            print(f"Error processing entity: {entity_error}")
                            continue

                    # Create relationships in Neo4j
                    for rel in relationships:
                        try:
                            source = str(rel.get('source', '')).replace("'", "''").replace('"', '""')
                            target = str(rel.get('target', '')).replace("'", "''").replace('"', '""')
                            relation_type = str(rel.get('type', 'RELATED')).replace("'", "''").replace('"', '""')
                            rel_description = str(rel.get('description', '')).replace("'", "''").replace('"', '""')

                            if source and target and len(source.strip()) > 0 and len(target.strip()) > 0:
                                query = f"""
                                MATCH (s:Entity {{name: '{source}'}})
                                MATCH (t:Entity {{name: '{target}'}})
                                MERGE (s)-[r:RELATED {{type: '{relation_type}', description: '{rel_description}', source_section: {section_idx}}}]->(t)
                                """
                                graph.query(query)
                                total_relationships += 1
                        except Exception as rel_error:
                            print(f"Error processing relationship: {rel_error}")
                            continue

                print(
                    f"✅ Section {section_idx + 1} processed: {len(entities)} entities, {len(relationships)} relationships")

            except Exception as section_error:
                print(f"Error processing section {section_idx + 1}: {section_error}")
                continue

    print("✅ Knowledge Graph successfully stored in Neo4j!")
    print(f"📊 Created approximately {total_entities} entities and {total_relationships} relationships")

    # Get actual counts from Neo4j
    try:
        entity_count = graph.query("MATCH (n:Entity) RETURN count(n) as count")[0]['count']
        relationship_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        print(f"📈 Final counts: {entity_count} entities and {relationship_count} relationships in Neo4j")
    except:
        pass

except Exception as e:
    print(f"⚠️ Error generating/storing knowledge graph: {e}")
    print("Continuing with vector-based retrieval only...")

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


# Function to retrieve information from Knowledge Graph
def retrieve_from_kg(query, graph, llm):
    """Retrieve relevant information from the knowledge graph"""
    try:
        # Create a GraphCypherQAChain for natural language queries
        kg_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )

        # Query the knowledge graph
        kg_result = kg_chain.invoke({"query": query})

        return {
            "answer": kg_result.get("result", ""),
            "cypher_query": kg_result.get("intermediate_steps", [{}])[0].get("query", ""),
            "context": kg_result.get("intermediate_steps", [{}])[0].get("context", "")
        }
    except Exception as e:
        print(f"⚠️ Error querying knowledge graph: {e}")
        return {"answer": "", "cypher_query": "", "context": ""}


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

# Retrieve documents from vector store
retrieved_docs = retrieval_rag_fusion.invoke({"question": question})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7)

# Retrieve information from Knowledge Graph
print("🔍 Querying Knowledge Graph...")
kg_info = retrieve_from_kg(question, graph, llm)

# Combine vector store context with knowledge graph context
vector_context = "\n".join(doc.page_content for doc in retrieved_docs)
kg_context = kg_info.get("answer", "")

# Enhanced Final Answer generation with both contexts
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

# Display retrieved context for debugging
print("\n📄 Retrieved Vector Context Chunks:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Chunk {i + 1} ---\n{doc.page_content}\n")

if kg_context:
    print("\n🕸️ Knowledge Graph Context:")
    print(kg_context)

if kg_info.get("cypher_query"):
    print(f"\n🔍 Generated Cypher Query:")
    print(kg_info["cypher_query"])
