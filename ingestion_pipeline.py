# ingestion_pipeline.py
import os
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load CSV
data_path = "data/Paytm.csv"
df = pd.read_csv(data_path)

# Convert rows to text (you can customize)
texts = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss_index.pkl")

# Save text mapping
with open("embeddings/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print(f"✅ Ingestion complete. Indexed {len(texts)} entries.")
