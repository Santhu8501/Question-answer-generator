# Question-answer-generator

Instructions to Read code:

I have developed a very basic skeleton model which has two methods to make question and answer generator using OpenAI llm and key related to openAI is in config.py
file which not provided in github


Method - 1 (General method) file name app.py :

In this general method I have created three functions which executes by taking input as pdf and output is posted directly into slack in the json format
function_1 : extracting_text - This function extracts text from the pdf path provided.
function_2 : answers_from_model - This function takes the text and list of questions and generate answers to the questions using gpt-3.5-turbo-0125
function_3 : data_to_slack - This function takes questions as well as answers as an input and post both the question and answer into the slack in json format

Method - 2 (implementing RAG) file name langchain.py :

I have created a basic simple version of RAG architecture and skeleton of architecture is given below
step -1 loading the pdf and extracting the text from the pdf
step -2 implementing recursive character text splitter in order to split the text into required chunk
step -3 initiating vectorDB as FAISS and by using openAI embeddings converting the chunks into embeddings and storing it in a vector store
step - 4 Making a retriever to retrieve the chunks related to given question

Due to Time constraint to connect to LLM I have given as a comments in langchain.py file so as the next step
step -5 Writing a prompt template in order to get the desired output
step - 6 initiating the openAI llm with required parameters like prompt template, model name, temperature etc....
step -7 by using langchain framework build a retrivalqa chain in order to chain llm and retriever
step -8 using chain.invoke method give your query and desired output is generated.

To know different methods to improve the accuracy of model please go through methodoly file.


