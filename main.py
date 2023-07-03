from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import streamlit as st
import json

model_name = "kaladin11/distilbert-base-uncased-finetuned-banking"

# import labels with json
with open("labels.json", "r") as f:
    labels = json.load(f)
num_labels = len(labels)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

st.text_input("Text here . . .", key="banking")

# prediction part
text = st.session_state.banking

if len(text) > 0:
    tokens = tokenizer(text, return_tensors="pt")

    # predictions
    output = model(**tokens)
    softmax = torch.nn.functional.softmax(output.logits, dim=-1)
    arg_max = torch.argmax(softmax, dim=1)
    predicted_label = labels[arg_max]
    st.text(f"Predicted label: {predicted_label}")
