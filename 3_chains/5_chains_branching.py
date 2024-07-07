
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_community.llms import Ollama

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "você é um assistente útil brasileiro."),
        ("human",
         "Gere uma nota em pt-br de agradecimento por este feedback positivo: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "você é um assistente útil brasileiro."),
        ("human",
         "Gere uma nota em pt-br de agradecimento por este feedback negativo: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "você é um assistente útil brasileiro."),
        (
            "human",
            "Gere uma nota em pt-br de agradecimento por este feedback neutro: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "você é um assistente útil brasileiro."),
        (
            "human",
            "Gere uma mensagem em pt-br para encaminhar esse feedback para um agente humano: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "você é um assistente útil brasileiro."),
        ("human",
         "Classifique o sentimento desse feedback como positivo, negativo, neutro ou crescente: {feedback}."),
    ]
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

# Run the chain with an example review
# Crítica boa - "O produto é excelente. Gostei muito de usá-lo e achei muito útil."
# Crítica ruim - "O produto é péssimo. Ele quebrou após apenas um uso e a qualidade é muito ruim."
# Avaliação neutra - "O produto está bom. Funciona conforme o esperado, mas nada de excepcional."
# Padrão - "Ainda não tenho certeza sobre o produto. Você pode me contar mais sobre seus recursos e benefícios?"

review = "O produto é terrível. Quebrou com apenas um uso e a qualidade é muito ruim."
result = chain.invoke({"feedback": review})

# Output the result
print(result)