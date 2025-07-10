import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline
import spacy
import spacy.cli  # 👈 Added for downloading model
import networkx as nx
import matplotlib.pyplot as plt

# ----- Ensure spaCy Model is Available -----
try:
    spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")

# ----- Set Page Config -----
st.set_page_config(page_title="Environmental AI Toolkit", layout="wide")

# ----- Models Setup -----
@st.cache_resource
def load_bert_pipeline():
    return pipeline("fill-mask", model="bert-base-uncased")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_image_generator():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    return pipe.to("cuda")

fill_mask = load_bert_pipeline()
nlp = load_spacy_model()

# ----- Sentence Classification -----
def classify_sentence(sentence):
    training_sentences = [
        "Solar panels help reduce carbon emissions.",
        "Forest conservation protects wildlife.",
        "Plastic waste pollutes oceans.",
        "Global warming increases sea levels.",
        "Recycling reduces landfill waste."
    ]
    labels = ['Renewable Energy', 'Conservation', 'Pollution', 'Climate Change', 'Waste Management']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(training_sentences)
    model = MultinomialNB()
    model.fit(X, labels)
    X_new = vectorizer.transform([sentence])
    return model.predict(X_new)[0]

# ----- NER Graph -----
def display_ner_graph(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    G = nx.Graph()
    for i, entity in enumerate(entities):
        G.add_node(entity)
        if i > 0:
            G.add_edge(entities[i-1], entity)
    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_color='lightgreen', font_weight='bold', ax=ax)
    st.pyplot(fig)

# ----- Image Generation -----
def generate_image(prompt):
    pipe = load_image_generator()
    image = pipe(prompt).images[0]
    st.image(image, caption="Generated Environmental Image")

# ----- Fill Mask -----
def fill_mask_output(masked_sentence):
    results = fill_mask(masked_sentence)
    for res in results:
        st.write(f"**Prediction:** {res['token_str']} | **Score:** {res['score']:.4f}")

# ----- Streamlit UI -----
st.title("🧠 Environmental AI Toolkit")

with st.sidebar:
    st.header("Choose a Task")
    task = st.radio("What do you want to do?", [
        "Classify a Sentence",
        "Generate an Environmental Image",
        "NER + Graph Mapping",
        "Fill-in-the-Blank Prediction"
    ])

# Task: Classification
if task == "Classify a Sentence":
    st.subheader("🔍 Sentence Classification")
    sentence = st.text_input("Enter an environmental sentence:")
    if sentence:
        result = classify_sentence(sentence)
        st.success(f"Predicted Category: **{result}**")

# Task: Image Generation
elif task == "Generate an Environmental Image":
    st.subheader("🖼️ Environmental Image Generator")
    prompt = st.text_input("Describe the scene you want:")
    if prompt:
        generate_image(prompt)

# Task: NER + Graph
elif task == "NER + Graph Mapping":
    st.subheader("🌐 Named Entity Recognition & Graph")
    text = st.text_area("Enter text related to environment projects:")
    if text:
        display_ner_graph(text)

# Task: Masked Fill-in-the-Blank
elif task == "Fill-in-the-Blank Prediction":
    st.subheader("🔤 Fill-in-the-Blank")
    masked_input = st.text_input("Type your sentence with a <mask> token:")
    if masked_input:
        fill_mask_output(masked_input)
