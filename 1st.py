from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv
import streamlit as st

template = """
You are a medical assistant. Based on the following symptoms and medical history, provide a list of possible conditions and suggest appropriate actions:

Symptoms: {symptoms}
Medical History: {medical_history}

Possible Conditions:
1. 
2. 
3. 

Suggested Actions:
1. 
2. 
3. 
"""

google_api_key = os.getenv('GOOGLE_API_KEY')
if google_api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please check your .env file.")
os.environ['GOOGLE_API_KEY'] = google_api_key

prompt = PromptTemplate(template=template, input_variables=["symptoms", "medical_history"])

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash') #gemini
memory = ConversationBufferWindowMemory(k=10)

chain = LLMChain(prompt=prompt, llm=llm)



st.title("Medical Assistant Chatbot")

input_symptoms = st.text_input("Enter the symptoms")
input_medical_history = st.text_input("Enter the medical history")

if st.button("Submit"):
    input_data = {
        "symptoms": input_symptoms,
        "medical_history": input_medical_history
    }
    response = chain.run(input_data)
    st.write(response)
