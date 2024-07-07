from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um comediante que conta piadas sobre {topic}."),
        ("human", "Conte me {piadas_count} piadas."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"topic": "programadores", "piadas_count": 3})

# Output
print(result)