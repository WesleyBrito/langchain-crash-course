
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
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

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "programadores", "piadas_count": 3})

# Output
print(response)