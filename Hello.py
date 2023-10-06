import streamlit as st
from transformers import AutoModelForSequenceClassification

# Load the Mini-LM model
model = AutoModelForSequenceClassification.from_pretrained("google/mini-lm-tf2-base")

# Define a function to generate legal advice
def generate_legal_advice(query):
    # Encode the query
    encoded_input = model.prepare_inputs_for_classification(query, return_tensors="tf")

    # Generate the legal advice
    legal_advice = model.generate(**encoded_input, max_length=100, num_beams=5)

    # Decode the legal advice
    legal_advice = model.decode(legal_advice, skip_special_tokens=True)

    return legal_advice

# Create a Streamlit app
st.title("AI Powered Legal Assistant")

# Get the user's query
query = st.text_input("Enter your legal question:")

# Generate the legal advice
legal_advice = generate_legal_advice(query)

# Display the legal advice
st.write(legal_advice)
