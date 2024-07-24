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


jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import re
import json
n_gpu_layers = 16
n_batch = 16
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
DB_FAISS_PATH = 'vectorstore/db_faiss'
# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = LlamaCpp(
        # cache=True,
        model_path="C:\\Users\\hari\\Phi-3-mini-4k-instruct-fp16.gguf",
        temperature=0,  # mistral-7b-instruct-v0.2.Q5_K_M.gguf
        top_p=1,
        n_ctx=2000,
        n_gpu_layers=n_gpu_layers,  # Phi-3-mini-4k-instruct-fp16.gguf
        n_batch=n_batch,
        use_mmap=True,
        streaming=False,
        use_mlock=True,  # force to keeo the model in ram
        max_tokens=1000,
        callback_manager=callback_manager,
        verbose=False)
    return llm
# Replace this with the path to your CSV file
csv_file_path = 'D:\\Program Files\\agent\\.venv\\Capitalaccountreceipts_0.csv'
loader = CSVLoader(file_path=csv_file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
db = FAISS.from_documents(data, embeddings)
# db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
history = []
while True:
    user_input = input("Query: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    result = chain({"question": user_input, "chat_history": history})
    history.append((user_input, result["answer"]))
    answer = result['answer']
    print(answer)
