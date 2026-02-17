import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("LangChain Ollama Chat App")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

llm = OllamaLLM(model="gemma:2b")  # ✅ Correct name

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

user_input = st.text_input("Ask something:")

if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
        st.write(response)