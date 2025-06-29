import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from dotenv import load_dotenv
from PIL import Image
import os
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wavfile
import speech_recognition as sr
import tempfile
import timm
import threading

# Load environment variables
load_dotenv()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Streamlit Page Config First
st.set_page_config(
    page_title="🩺 Medical Chatbot Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Beautification ---
st.markdown(
    """
    <style>
    /* Entire App Background and Font */
    body, .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
        color: #2c2c2c;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111; /* True Dark Sidebar */
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    section[data-testid="stSidebar"] ul {
        color: white !important;
        list-style-type: none;
        padding-left: 1em;
    }
    section[data-testid="stSidebar"] ul li::before {
        content: "• ";
        color: #4CAF50;  /* Green dot for bullets */
    }
    section[data-testid="stSidebar"] ul li {
        margin-bottom: 10px;
        font-size: 16px;
        color: white !important;
    }

    /* Title, Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c2c2c;
    }

    /* Input box */
    .stTextInput>div>div>input {
    color: #000000 !important; /* Actual text color */
    background-color: #ffffff;
    border: 1px solid #cccccc;
    border-radius: 8px;
    padding: 12px;
    font-size: 16px;
}
    input::placeholder,
textarea::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #222222 !important; /* Dark near-black */
    font-size: 16px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
    /* File uploader */
    .stFileUploader > div > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }

    /* Button */
    .stButton>button {
        background-color: #6200EA;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        margin-top: 10px;
        font-weight: bold;
    }

    /* Chat bubbles */
    .chat-bubble-user {
        background-color: #d0f0fd;
        border-radius: 12px 12px 0px 12px;
        padding: 12px 20px;
        margin-bottom: 15px;
        text-align: left;
        width: fit-content;
        max-width: 80%;
        color: #000000;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .chat-bubble-bot {
        background-color: #d0f0fd ;
        border-radius: 12px 12px 12px 0px;
        padding: 12px 20px;
        margin-bottom: 15px;
        text-align: left;
        width: fit-content;
        max-width: 80%;
        color: #000000;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }

    .stMarkdown {
        color: #2c2c2c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Hugging Face Model for Text QA
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Load FAISS Vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load Pretrained Vision Model
@st.cache_resource
def load_medical_model():
    model = timm.create_model('densenet121', pretrained=True)
    model.classifier = torch.nn.Linear(1024, 14)
    model = model.to(device)
    model.eval()
    return model

model = load_medical_model()

# Disease Labels
disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
                  'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                  'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# Preprocessing for Images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_text_sync(text):
    def run_tts():
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        local_engine.say(text)
        local_engine.runAndWait()
    threading.Thread(target=run_tts).start()

# Predict Disease
def predict_disease(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    results = {disease: prob for disease, prob in zip(disease_labels, probs)}
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

# Record audio from microphone
def record_audio(duration=5, fs=16000):
    st.info(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return fs, audio

# Recognize audio
def recognize_from_audio(fs, audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wavfile.write(f.name, fs, audio)
        recognizer = sr.Recognizer()
        with sr.AudioFile(f.name) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Speech recognition service unavailable."

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/doctor-male.png", width=100)
    st.title("🩺 Medical Chatbot")
    st.markdown("""
    <b>AI-Powered Medical Assistant</b><br><br>
    <ul>
        <li>📚 <b>Text-based Medical QA</b></li>
        <li>🖼️ <b>X-ray Image Analysis</b></li>
        <li>🎙️ <b>Voice to Text</b></li>
    </ul>
    """, unsafe_allow_html=True)
    speak = st.radio("🔈 Bot Voice:", ("🔊 Speak Answers", "🔇 Mute"))

# Main UI
st.title("🩺 Medical Assistant Chatbot")
st.markdown("<h4 style='text-align: center; color: grey;'>Ask medical questions, upload X-rays, or talk live! 🎙️</h4>", unsafe_allow_html=True)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload Medical Image
st.subheader("📤 Upload a Chest X-ray Image (jpg, png)")
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

image_diagnosis = ""
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    with st.spinner("🧠 Analyzing X-ray..."):
        predictions = predict_disease(image)

    st.subheader("📝 X-ray Diagnosis (Top 5 Conditions):")
    for disease, prob in predictions[:5]:
        st.write(f"**{disease}**: {prob*100:.2f}%")

    diagnosis_summary = ", ".join([f"{disease} ({prob*100:.1f}%)" for disease, prob in predictions[:3]])
    image_diagnosis = f"Top findings: {diagnosis_summary}."

    if speak == "🔊 Speak Answers":
        speak_text_sync(f"The uploaded X-ray analysis: {diagnosis_summary}")

# Upload Voice Input
st.subheader("🎤 Upload a Voice Question (WAV file)")
uploaded_audio = st.file_uploader("Upload your voice question (WAV)", type=["wav"])

user_question = st.text_input("🔎 Or Type your medical question:")

# Handle Uploaded Audio
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")
    st.write("🎙️ Recognizing Uploaded Speech...")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_audio.getbuffer())
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
    try:
        recognized_text = recognizer.recognize_google(audio_data)
        st.success(f"You said: {recognized_text}")
        user_question = recognized_text
    except:
        st.error("Speech Recognition failed!")

# Record Live Audio
st.subheader("🎙️ Live Microphone Input (Record)")
if st.button("🎤 Start Recording (5 sec)"):
    fs, audio = record_audio(duration=5)
    recognized_text = recognize_from_audio(fs, audio)
    st.success(f"You said: {recognized_text}")
    user_question = recognized_text

# Answer Generation
if user_question:
    with st.spinner('🤖 Generating Answer...'):
        retriever = vectorstore.as_retriever()
        context_docs = retriever.get_relevant_documents(user_question)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        prompt = f"""
You are a highly knowledgeable and detailed medical assistant.
Please provide a comprehensive and elaborate answer to the following question using the provided context.

Image Diagnosis (if any): {image_diagnosis}

Context:
{context_text}

Question: {user_question}

Answer:
"""
        result = qa_pipeline(prompt, max_length=512, min_length=150)[0]['generated_text']

        if speak == "🔊 Speak Answers":
            speak_text_sync(result)

        st.session_state.chat_history.append((user_question, result))

# Show Chat History
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="chat-bubble-user">🧑‍⚕️ <strong>You:</strong> {q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-bot">🤖 <strong>Bot:</strong> {a}</div>', unsafe_allow_html=True)

