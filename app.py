import streamlit as st
from embeddings import generate_embedding
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="AI Code Error Explainer", layout="centered")

st.title("🤖 AI Code Error Explainer")
st.write("Paste your programming error in simple English and get smart explanations.")

# ---------- USER INPUT (UI FIRST!) ----------
user_error = st.text_area(
    "Enter your error message 👇",
    placeholder="Example: my program crashes when accessing list index"
)

# ---------- BUTTON ----------
if st.button("Explain Error"):
    if user_error.strip() == "":
        st.warning("⚠️ Please enter an error message.")
    else:
        with st.spinner("Analyzing error using AI..."):
            # Load data only when needed
            with open("data/errors.txt", "r") as file:
                errors = file.read().split("\n\n")

            vector_db = []
            for error in errors:
                vector_db.append({
                    "vector": generate_embedding(error),
                    "text": error
                })

            query_vector = generate_embedding(user_error)

            similarities = []
            for item in vector_db:
                score = cosine_similarity(
                    [query_vector], [item["vector"]]
                )[0][0]
                similarities.append((score, item["text"]))

            similarities.sort(reverse=True, key=lambda x: x[0])

        st.subheader("🔍 Top 3 Similar Errors")
        for i, (score, text) in enumerate(similarities[:3], start=1):
            st.markdown(f"### Match {i} (Similarity: {score:.2f})")
            st.code(text)
