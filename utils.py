import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

# Neo4j connection setup
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphdbms"

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

class FreeKGGenerator:
    def __init__(self, api_key=None, model='gemini-2.0-flash', temperature=0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key or os.environ.get("GOOGLE_API_KEY")
        )
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
        try:
            chain = self.kg_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"text": text})
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            try:
                kg_data = json.loads(response)
                return kg_data
            except json.JSONDecodeError:
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

def retrieve_from_kg(query, llm):
    try:
        kg_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )
        kg_result = kg_chain.invoke({"query": query})
        return {
            "answer": kg_result.get("result", ""),
            "cypher_query": kg_result.get("intermediate_steps", [{}])[0].get("query", ""),
            "context": kg_result.get("intermediate_steps", [{}])[0].get("context", "")
        }
    except Exception as e:
        print(f"⚠️ Error querying knowledge graph: {e}")
        return {"answer": "", "cypher_query": "", "context": ""}