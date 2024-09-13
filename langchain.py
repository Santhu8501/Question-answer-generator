from langchain_community.document_loaders import PyPDFLoader
import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from config import config

openai.api_key = config.openAI_api_key


# loading the pdf loader
def pdf_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = ''
    for page in pages:
        text += page.page_content
    return text

# initiating the function
pdf_path = 'Give your pdf path'
data = pdf_loader(pdf_path)


# using text splitter to split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size= 800, chunk_overlap= 100)
text_split = text_splitter.split_text(data)


# intiating the vector store to store embeddings
db = FAISS.from_texts(text_split, OpenAIEmbeddings())
retriver = db.as_retriever(search_kwargs={"k": 5})


# retriving the context to check weather data related to question or not
res = retriver.invoke('list the potential reasonable accommodations for pregnant employees?')
print(res)

''' then give this res from the retriver as an input to the function answers_from_model(text, questions) in app.py file 
so output answers are generated using openai llm and then use the function data_to_slack(channel, json_data) in app.py to connect to the webclient slack.
'''


