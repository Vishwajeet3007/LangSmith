from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_PROJECT"]="Sequential LLM Application"
load_dotenv()

# Step 1 : Detailed report prompt
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

# Step 1: Summary prompt
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=['text']
)

# Models 
model_pro = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser = StrOutputParser()

# Chain : topic -> detailed report(pro) -> summary(flash) -> parser
chain = prompt1 | model_pro | parser | prompt2 | model_flash | parser

config ={
    'run_name':"Sequentail chain", # For change default name Runnable sequence to Sequential chain
    "tags":['llm application', 'sequential chain', 'report generation','Summarization'],
    "metadata":{'model1':"gemini-1.5-flash", 'model2':"gemini-1.5-flash", 'prompt1':'Detailed report', 'prompt2':'Summary'},
}
# Run the chain
result = chain.invoke({"topic":"Unemployment in India"},config=config)
print(result)