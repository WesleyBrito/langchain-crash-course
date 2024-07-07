
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_community.llms import Ollama

# Create a Ollama model
model = Ollama(base_url="http://host.docker.internal:11434", model="llama3")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é brasileiro, revisor especialista em produtos."),
        ("human", "Liste as principais características do produto {nomeProduto}."),
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é brasileiro, revisor especialista em produtos."),
            (
                "human",
                "Dadas essas características: {features}, liste os prós e/ou beneficios desses recursos.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Você é brasileiro, revisor especialista em produtos."),
            (
                "human",
                "Dadas essas características: {features}, liste os contras desses recursos.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"Pros": pros_branch_chain, "Contras": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"nomeProduto": "MacBook Pro"})

# Output
print(result)
