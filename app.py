import streamlit as st
import joblib

# -------------------------------
# Load model & vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("lang_nb_model.pkl")
    vectorizer = joblib.load("lang_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()


# -------------------------------
# Detection logic
# -------------------------------
def detect_possible_languages(
    text,
    threshold=0.25,
    min_len=3
):
    words = text.split()

    X = vectorizer.transform(words)
    probs = model.predict_proba(X)
    classes = model.classes_

    results = []

    for i, word in enumerate(words):
        # Reject very short or non-alphabetic tokens
        if len(word) < min_len or not word.isalpha():
            results.append((word, ["Unknown"]))
            continue

        possible_langs = [
            lang
            for lang, p in zip(classes, probs[i])
            if p >= threshold
        ]

        if not possible_langs:
            possible_langs = ["Unknown"]

        results.append((word, possible_langs))

    return results


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Multilingual Language Analyzer",
    layout="centered"
)

st.title("🌍 Multilingual Language Analyzer")
st.write(
    "This app analyzes each word and outputs **all possible languages** "
    "based on character-level probabilities. "
    "Words that do not match any learned language are labeled **Unknown**."
)

text_input = st.text_input(
    "Enter text",
    placeholder="Saya is naive bayes roeireirpepre"
)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = detect_possible_languages(text_input)

        st.subheader("Results")

        for word, langs in results:
            st.code(
                f"{word:<15} → {', '.join(langs)}",
                language="text"
            )

st.caption(
    "Model: Character-level Naive Bayes | Output reflects linguistic ambiguity"
)