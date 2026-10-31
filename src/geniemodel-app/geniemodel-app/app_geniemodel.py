import os
import streamlit as st
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints"
)

MODEL_NAME = "databricks-gpt-oss-20b"
st.title("AI Assistant")
question = st.text_input(
    "Ask a question"
)


def ask_model(question):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"
    


if st.button("Ask"):
    try:
        with st.spinner("Thinking..."):
            answer = ask_model(question)
        st.success("Response")
        st.write(answer)
    except Exception:
        st.error(
            "The AI service is currently unavailable. Please try again later."
        )