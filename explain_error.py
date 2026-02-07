from embeddings import generate_embedding
from sklearn.metrics.pairwise import cosine_similarity

def explain_error():
    vector_db = []

    with open("data/errors.txt", "r") as file:
        errors = file.read().split("\n\n")

    for error in errors:
        vector_db.append({
            "vector": generate_embedding(error),
            "text": error
        })

    user_error = input("Paste your error message: ")
    query_vector = generate_embedding(user_error)

    similarities = []
    for item in vector_db:
        score = cosine_similarity([query_vector], [item["vector"]])[0][0]
        similarities.append((score, item["text"]))

    similarities.sort(reverse=True, key=lambda x: x[0])

    print("\n🔍 Top 3 similar errors:\n")
    for i, (score, text) in enumerate(similarities[:3], start=1):
        print(f"--- Match {i} (Similarity: {score:.2f}) ---")
        print(text)
        print()

if __name__ == "__main__":
    explain_error()
