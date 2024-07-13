import langchain
from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
import openai
import os
import getpass
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from PyPDF2 import PdfReader
import json
import config
from config import config



# extracting the text from pdf
def extracting_text(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            # print(page_num)
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e :
        return str(e)


# defining the api_key securely
openai.api_key = config.openAI_api_key

#intiating the model and invoking with text and qquestions as input
def answers_from_model(text, questions):
    try:
        qa_pairs = []
        for question in questions:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. You need to answer the question using below text, if you do not know the answer don't makeup the answer just reply as Data Not Available"},
                    {"role": "user", "content": f"Context: {text}"},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                max_tokens=200
            )
            answer = result.choices[0].text.strip()
            qa_pairs.append({'Question' : question, 'Answer' : answer})
            return qa_pairs
    except Exception as e:
        return str(e)




client = WebClient(token= config.Slack_token)

# posting data to slack in specified channel in json format
def data_to_slack(channel, json_data):
    try:
        for message in json_data:
            client.chat_postMessage(channel=channel, text=message)
    except SlackApiError as e:
        print(f"Slack API Error: {str(e)}")



# definig a main method which invokes all the three defined functions
def main(pdf_path, questions, channel):
    text = extracting_text(pdf_path)
    Q_and_A = answers_from_model(text, questions)
    json_data = json.dumps(Q_and_A, indent=2)
    data_to_slack(channel, json_data)



# initialising by main method
if __name__ == '__main__':
    pdf_path = 'Path of pdf'
    questions = ['list of questions']
    channel = 'your slack channel'
    main(pdf_path, questions, channel)












