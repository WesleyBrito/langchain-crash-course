from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")


chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="Você é um assistente de IA útil e se comunicará em pt-br sempre.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("Você: ")
    if query.lower() == "sair":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Historico das mensagens ----")
print(chat_history)