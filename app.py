from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant Cloud using env vars
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "product_texts"

def init_collection():
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

init_collection()

@app.route("/store-product-text", methods=["POST"])
def store_product_text():
    try:
        raw_text = request.data.decode("utf-8").strip()
        if not raw_text:
            return jsonify({"error": "No text received"}), 400

        vector = model.encode(raw_text).tolist()

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": raw_text}
        )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        return jsonify({"message": "✅ Product stored in Qdrant Cloud"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)
