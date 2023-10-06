import streamlit as st
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, device_map="auto", torch_dtype=torch.float32
)
embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
db = Chroma(persist_directory="ipc_vector_data", embedding_function=embeddings)
pipe = pipeline(
    "text2text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.95,
)
local_llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)

chat_history = []


def chat():
    """Returns a response to the user input."""
    user_input = st.session_state["input_query"]
    bot_response = qa_chain({"query": user_input})
    bot_response = bot_response["result"]
    response = ""
    for letter in "".join(bot_response):
        response += letter + ""
    chat_history.append((user_input, response))
    st.session_state["chat_history"] = chat_history
    st.session_state["input_query"] = ""
    return response


st.title("Chatbot")

st.session_state["chat_history"] = chat_history
st.session_state["input_query"] = ""

st.text("Chat history:")
st.write(st.session_state["chat_history"])

input_query_element = st.text_area("Input", value=st.session_state["input_query"])
st.button("Submit", on_click=chat)

if __name__ == "__main__":
    st.run()
