# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# Ollama Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/ollama/

from dotenv import load_dotenv
from langchain_community.llms import Ollama

# Load environment variables from .env
load_dotenv()

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# Invoke the model with a message
result = model.invoke("Quanto é 81 dividido por 9?")
print("Resultado:")
print(result)