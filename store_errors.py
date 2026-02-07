from embeddings import generate_embedding

# Simple in-memory vector store (simulating Endee usage)
vector_db = []

def store_errors():
    with open("data/errors.txt", "r") as file:
        errors = file.read().split("\n\n")

    for error in errors:
        vector = generate_embedding(error)
        vector_db.append({
            "vector": vector,
            "text": error
        })

    print("✅ Errors embedded and stored using Endee-style vector storage")

if __name__ == "__main__":
    store_errors()
