import os
import json
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_neo4j import Neo4jGraph
from utils import FreeKGGenerator

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


def setup_data(pdf_path="./YOUR PATH", persist_directory="./chroma_db"):
    # Initialize the free KG generator
    kg = FreeKGGenerator(api_key=os.environ["GOOGLE_API_KEY"])

    # Load PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Create hierarchical chunks
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n\n", "\n\n", "\n", " ", ""]
    )
    sections = section_splitter.split_documents(documents)

    # Paragraph splitter
    paragraph_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    # Store chunks
    all_chunks = []
    kg_texts = []

    for section_idx, section in enumerate(sections):
        section_id = f"section_{section_idx}"
        section_text = section.page_content
        lines = section_text.strip().split("\n")
        section_title = f"Section {section_idx}"
        if lines and len(lines[0]) < 100:
            section_title = lines[0]

        paragraphs = paragraph_splitter.create_documents(
            [section_text],
            metadatas=[section.metadata]
        )

        for para_idx, para_doc in enumerate(paragraphs):
            all_chunks.append({
                "text": para_doc.page_content,
                "is_small_chunk": "True",
                "chunk_id": f"{section_id}_para_{para_idx}",
                "parent_id": section_id,
                "parent_title": section_title,
                "section_index": str(section_idx),
                "paragraph_index": str(para_idx),
                "page": str(section.metadata.get("page", "0"))
            })

        all_chunks.append({
            "text": section_text,
            "is_small_chunk": "False",
            "chunk_id": section_id,
            "parent_id": "",
            "section_title": section_title,
            "section_index": str(section_idx),
            "page": str(section.metadata.get("page", "0"))
        })
        kg_texts.append(section_text)

    # Generate and store Knowledge Graph
    print("🔄 Generating Knowledge Graph...")
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        batch_size = 3
        total_entities = 0
        total_relationships = 0

        for batch_start in range(0, len(kg_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(kg_texts))
            batch_texts = kg_texts[batch_start:batch_end]
            print(f"Processing batch {batch_start // batch_size + 1}/{(len(kg_texts) - 1) // batch_size + 1}...")

            for i, text in enumerate(batch_texts):
                section_idx = batch_start + i
                if len(text) > 3000:
                    text = text[:3000] + "..."
                kg_result = kg.generate(text)
                if kg_result and isinstance(kg_result, dict):
                    entities = kg_result.get('entities', [])
                    relationships = kg_result.get('relationships', [])
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

        print(f"✅ Knowledge Graph stored in Neo4j: {total_entities} entities, {total_relationships} relationships")
        try:
            entity_count = graph.query("MATCH (n:Entity) RETURN count(n) as count")[0]['count']
            relationship_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
            print(f"📈 Final counts: {entity_count} entities and {relationship_count} relationships in Neo4j")
        except:
            pass
    except Exception as e:
        print(f"⚠️ Error generating/storing knowledge graph: {e}")

    # Create embeddings and store in ChromaDB
    texts = [chunk["text"] for chunk in all_chunks]

    def clean_metadata(metadata_dict):
        cleaned = {}
        for k, v in metadata_dict.items():
            if k != "text":
                if v is None:
                    v = ""
                if not isinstance(v, (bool, int, float, str)):
                    v = str(v)
                cleaned[k] = v
        return cleaned

    metadatas = [clean_metadata(chunk) for chunk in all_chunks]
    ids = [chunk["chunk_id"] for chunk in all_chunks]
    embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ["GOOGLE_API_KEY"])
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embed,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print(f"✅ Vector store created and persisted at {persist_directory}")


if __name__ == "__main__":
    setup_data()
