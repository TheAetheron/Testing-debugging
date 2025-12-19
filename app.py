import streamlit as st
import pandas as pd
import joblib
import altair as alt

model = joblib.load("lang_nb_model.pkl")
vectorizer = joblib.load("lang_vectorizer.pkl")
classes = model.classes_

def confidence_label(p):
    if p >= 0.75:
        return "High"
    elif p >= 0.40:
        return "Medium"
    else:
        return "Low"


def is_ambiguous(word_probs, margin=0.10):
    if len(word_probs) < 2:
        return False
    return abs(word_probs[0][1] - word_probs[1][1]) < margin


def detect_words(text, threshold=0.10, unknown_threshold=0.25):
    words = [w.strip() for w in text.split() if w.strip()]
    if not words:
        return []

    X = vectorizer.transform(words)
    probs = model.predict_proba(X)

    results = []

    for i, word in enumerate(words):
        word_probs = [
            (lang, float(p))
            for lang, p in zip(classes, probs[i])
            if p >= threshold
        ]

        word_probs.sort(key=lambda x: x[1], reverse=True)

        if not word_probs or word_probs[0][1] < unknown_threshold:
            results.append((word, [("Unknown", 1.0)]))
        else:
            results.append((word, word_probs))

    return results

st.set_page_config(page_title="Language Detection NLP", layout="centered")

st.title("🌍 Multilingual Word Language Detection")

st.markdown(
    """
This application detects the **most likely language of each word**
using a **character n-gram Naive Bayes model**.

"""
)

user_input = st.text_area(
    "Enter words (space or newline separated):",
    placeholder="i eat にちは sapi",
    height=120
)

predict_clicked = st.button("🔍 Detect")

if user_input:
    results = detect_words(user_input)

    if not results:
        st.warning("No valid input detected.")
    else:
        known_count = 0
        unknown_count = 0

        st.markdown("## 🔍 Analysis Results")

        for word, probs in results:
            with st.expander(f"Word: **{word}**"):
                if probs[0][0] == "Unknown":
                    st.error("❌ Unknown word — low confidence prediction")
                    unknown_count += 1
                else:
                    top_lang, top_prob = probs[0]
                    known_count += 1

                    st.write(f"**Prediction:** {top_lang}")
                    st.write(
                        f"**Confidence:** {top_prob:.2f} "
                        f"({confidence_label(top_prob)})"
                    )

                    if is_ambiguous(probs):
                        st.warning("⚠️ Ambiguous prediction — multiple languages likely")

                    df_probs = pd.DataFrame(
                        probs[:5],
                        columns=["Language", "Probability"]
                    )
                    chart = (
                        alt.Chart(df_probs)
                        .mark_bar()
                        .encode(
                             y=alt.Y(
                                "Language:N",
                                sort="-x",                     # highest prob on top
                                axis=alt.Axis(title=0)    # normal horizontal labels
                            ),
                            x=alt.X(
                                "Probability:Q",
                                scale=alt.Axis(title="confidence", format=".2f") # probability scale
                            ),
                            tooltip=["Language", "Probability"]
                        )
                        .properties(height=120)
                    )

                    
                    st.altair_chart(chart, use_container_width=True)

        st.markdown("## 📊 Summary")
        st.write(f"Known words: **{known_count}**")
        st.write(f"Unknown words: **{unknown_count}**")

st.write(
    """
### Known limitations
- Non-words may still receive low probabilities
- Similar languages can be ambiguous
- Word-level detection lacks sentence context
"""
)

st.markdown("### 🌐 Supported Languages")
st.write(", ".join(sorted(classes)))
