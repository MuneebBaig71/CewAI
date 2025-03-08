from crewai import Agent, Task, Crew, Process, LLM
import os
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Ensure the GEMINI_API_KEY is loaded
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")

# Create a knowledge source
content = "Users name is John. He is 30 years old and lives in San Francisco."
text_source = TextFileKnowledgeSource(
    file_paths=['data1.txt'],
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gemini/gemini-1.5-flash", temperature=0, api_key=GEMINI_API_KEY)

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="""You are a master at understanding people and their preferences.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    # knowledge_sources=[text_source]
)

task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[text_source] # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
)

def apply():
    result = crew.kickoff(inputs={"question": "What city does John live in and how old is he?"})
    print(result)

apply()