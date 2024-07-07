from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Resolva os problemas matematicos abaixo em pt-br"),
    HumanMessage(content="Quanto é 81 dividido por 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Resposta da AI: {result}")

# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Resolva os problemas matematicos abaixo em pt-br"),
    HumanMessage(content="Quanto é 81 dividido por 9?"),
    AIMessage(content="81 dividido por 9 é 9."),
    HumanMessage(content="Quanto é 10 vezes 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Resposta da AI: {result}")