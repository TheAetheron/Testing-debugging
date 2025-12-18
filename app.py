import streamlit as st
import joblib
import pandas as pd

# Load model & vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("lang_nb_model.pkl")
    vectorizer = joblib.load("lang_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()


def smooth_predictions(results, strong_threshold=0.6, weak_threshold=0.05):
    raw_preds = []
    for token, lang_probs in results:
        if lang_probs:
            raw_preds.append(lang_probs[0][0])
        else:
            raw_preds.append(None)

    final_preds = []

    for i, (token, lang_probs) in enumerate(results):
        if lang_probs and lang_probs[0][1] >= strong_threshold:
            final_preds.append(lang_probs[0][0])
            continue

        neighbor_langs = []
        if i > 0:
            neighbor_langs.append(raw_preds[i - 1])
        if i < len(results) - 1:
            neighbor_langs.append(raw_preds[i + 1])

        neighbor_langs = [x for x in neighbor_langs if x is not None]

        resolved = None
        for nlang in neighbor_langs:
            for lang, prob in lang_probs:
                if lang == nlang and prob >= weak_threshold:
                    resolved = lang
                    break
            if resolved:
                break

        final_preds.append(resolved if resolved else lang_probs[0][0])

    return list(zip([t for t, _ in results], final_preds))


def detect_words(text, threshold=0.10):
    words = text.split()
    X = vectorizer.transform(words)
    probs = model.predict_proba(X)
    classes = model.classes_

    results = []
    for i, word in enumerate(words):
        word_probs = []
        for lang, p in zip(classes, probs[i]):
            if p >= threshold:
                word_probs.append((lang, float(p)))
        word_probs.sort(key=lambda x: x[1], reverse=True)
        results.append((word, word_probs))

    return results


def detect_with_smoothing(text):
    results = detect_words(text)
    return smooth_predictions(results)


# ---------------- UI ----------------

st.set_page_config(page_title="Multilingual Language Detection", layout="centered")

st.title("🌍 Multilingual Language Detection")
st.write("Detect language **per word** using a Naive Bayes model.")

text_input = st.text_input(
    "Enter text",
    placeholder="i eat にちは sapi"
)

if st.button("Detect Language"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        output = detect_with_smoothing(text_input)

        st.subheader("Detection Result")

        # Display exactly like console output
        for token, lang in output:
            st.code(f"{token:<10} → {lang}", language="text")
