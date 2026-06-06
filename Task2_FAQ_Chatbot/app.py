import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq = pd.read_csv("faq.csv")

questions = faq["Question"]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

st.title("🤖 FAQ Chatbot")

user_question = st.text_input("Ask a Question")

if st.button("Get Answer"):

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    index = similarity.argmax()

    if similarity.max() < 0.3:
        st.error("Sorry, I couldn't find a suitable answer.")
    else:
        answer = faq.iloc[index]["Answer"]
        st.success(answer)