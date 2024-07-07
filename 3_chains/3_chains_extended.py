
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_community.llms import Ollama

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um comediante que conta piadas em pt-br sobre {topic}."),
        ("human", "Diga-me {piadas_count} piadas."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Quantidade de palavras: {len(x.split())}\n{x}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "programadores", "piadas_count": 3})

# Output
print(result)