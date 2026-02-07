from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text: str):
    """
    Converts text into a vector embedding
    """
    return model.encode(text).tolist()
