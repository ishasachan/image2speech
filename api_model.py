#used hugging face model - serverless interface for prototyping 

#image to text 
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer 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"}

def image_to_text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# output = image_to_text("img.jpeg")
# print(output)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_messages(
      [
        ("system" , "YOu are a amazing story teller . Please generate a 100 words story based on the given scenario"),
        ("user", "input:{input}")
      ]
)

llm=ChatOpenAI(model='gpt-3.5-turbo')
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

# story=chain.invoke({'input': output})
# print(story)

#text to speech
def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payload={
        "inputs":message
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    with open('story.mp3' , 'wb')as file:
        file.write(response.content)
    # return response.json()

# text_to_speech("there is a man and woman hugging each other in mountains . the girl is so beuatiful but boy is ugly. thank you for this bye good night have a nice day")
# text_to_speech(story)

#ui
import streamlit as st
def main():
    st.header('Image to Story Generator')
    uploaded_file=st.file_uploader("Choose an image" , type='jpg')

    if uploaded_file is not None:
        bytes_data= uploaded_file.getvalue()
        with open(uploaded_file.name , "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file , caption='Uploaded Image',
                 use_column_width=True)
        output = image_to_text(uploaded_file.name)
        story=chain.invoke({'input': output})
        text_to_speech(story)

        with st.expander("Scenario"):
            st.write(output[0]['generated_text'])
        with st.expander("story"):
            st.write(story)

        st.audio("story.mp3")


if __name__ == '__main__':
    main()