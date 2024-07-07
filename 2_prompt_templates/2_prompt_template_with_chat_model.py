from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# PART 1: Create a ChatPromptTemplate using a template string
print("-----Prompt do Template-----")
template = "Me conte uma piada em pt-br sobre {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "gatos"})
result = model.invoke(prompt)
print(result)

# PART 2: Prompt with Multiple Placeholders
print("\n----- Start Prompt com multiplos placeholders -----\n")
template_multiple = """Você é um assistente útil e conversa sempre em pt-br.
Human: Diga-me um {adjective} breve historia sobre {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "engraçado", "animal": "panda"})

result = model.invoke(prompt)
print(result)
print("\n----- End Prompt com multiplos placeholders -----\n")

# PART 3: Prompt with System and Human Messages (Using Tuples)
print("\n----- Start Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "Você é um comediante que conta piadas sobre {topic} em pt-br."),
    ("human", "Diga-me {joke_count} piadas."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "programadores", "joke_count": 3})
result = model.invoke(prompt)
print(result)
print("\n----- End Prompt with System and Human Messages (Tuple) -----\n")