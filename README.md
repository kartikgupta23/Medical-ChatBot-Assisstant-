# 🩺 Medical Chatbot Assistant

A powerful AI-powered medical chatbot that answers medical questions, analyzes X-ray images, and supports voice queries. Built with LLaMA/LLM models, Hugging Face embeddings, FAISS, Pinecone, and Streamlit/Flask.

---

## 🚀 Features
- **Text-based Medical Q&A**
- **X-ray Image Analysis**
- **Voice-to-Text Medical Queries**
- **Modern Web UI (Streamlit/Flask)**
- **Uses LLMs and Embeddings for Contextual Answers**

---

## 🛠️ Setup & Installation (Python 3.11+)

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd MedicalChatbot/MedicalChatbot
```

### 2. Create and Activate a Virtual Environment
```sh
python -m venv venv
./venv/Scripts/activate  # On Windows
```

### 3. Install Dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the `MedicalChatbot/MedicalChatbot` directory with your API keys:
```

HF_TOKEN=your_huggingface_token
```

### 5. Prepare the Index (if needed)
If `faiss_index/index.faiss` and `index.pkl` are missing, run:
```sh
python store_index.py
```

### 6. Run the Application
For the Streamlit/voice/image chatbot:
```sh
python app.py
```
For the Flask/Pinecone chatbot:
```sh
python medical_chatbot.py
```

### 7. Open in Browser
Go to the address shown in the terminal (usually http://127.0.0.1:5000/ or as shown by Streamlit).

---

## 📹 Video Tutorial

A step-by-step video guide (screen recording) for setup and usage will be available soon!

**The video will be uploaded directly to this GitHub repository (as a release asset or in the repo).**

*The video will be added here once available. Want it sooner or have suggestions? [Open an Issue](../../issues) to request or track the video tutorial!*

---

## 📄 License
MIT License. See [LICENSE](./LICENSE).

## 🤝 Contributing
Pull requests and issues are welcome!

## 📧 Contact
For questions, open an issue or email me directly,
Email: kartikeyagupta1435@gmail.com
Linkedln: linkedln.com/kartikeyagupta19


