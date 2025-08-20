from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Laod_API keys from.env file
load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# model initialization
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# output parser
parser = StrOutputParser()

# chain:prompt -> model -> parser
chain = prompt | model | parser

# Run the chain with a question
result = chain.invoke({"question":"What is th capital of India?"})
print(result)
