from crewai import Crew, Process, Agent, Task, LLM
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from typing import List, Optional
import os

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

text_source = TextFileKnowledgeSource(
    file_path=['user_preference.txt'],
)
gemini_llm = LLM(
    model="gemini/gemini-1.5-pro-002",
    api_key=GEMINI_API_KEY,
    temperature=0,
)

agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="""You are a master at understanding people and their preferences.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm,
)

task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

# Assemble your crew with memory capabilities
crew = Crew(
    agents=[agent],
    tasks=[task],
    knowledge_sources=[text_source],
    process=Process.sequential,
    embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": GEMINI_API_KEY,
                }
            },
    memory=True,
    # Long-term memory for persistent storage across sessions
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path="./my_crew1/long_term_memory_storage.db"
        )
    ),
    # Short-term memory for current context using RAG
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": GEMINI_API_KEY,
                }
            },
            type="short_term",
            path="./my_crew1/short_term1/"
        )
    ),
    verbose=True,
    entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": GEMINI_API_KEY,
                }
            },
            type="entity",
            path="/my_crew1/"
        )
    ),
)

def deploy():
    run = crew.kickoff(inputs={'question':'Where does the user live?'})
    print(run)

