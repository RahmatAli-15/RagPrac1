# retrieval_pipeline.py
import os
import faiss
import pickle
import numpy as np
import speech_recognition as sr
import pyttsx3
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Load components
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss_index.pkl")

with open("embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
engine = pyttsx3.init()

def record_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print(f"🗣️ You said: {query}")
        return query
    except:
        print("❌ Could not understand audio")
        return None

def retrieve_context(query, top_k=3):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)
    return "\n".join([texts[i] for i in I[0]])

def speak(text):
    engine.say(text)
    engine.runAndWait()

def chat():
    query = record_voice()
    if not query:
        return
    context = retrieve_context(query)
    messages = [
        {"role": "system", "content": "You are an expert expense manager assistant. Use provided context to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]
    answer = llm.invoke(messages)
    print(f"🤖 Answer: {answer.content}")
    speak(answer.content)

if __name__ == "__main__":
    chat()
