import sys
import streamlit as st

sys.path.append("../scripts")

from utils import load_sa_model, generate_prediction, load_wv_model

sa_model = load_sa_model()
wv_model = load_wv_model()

st.header("Sentimental Analyzer with Word2Vec!")
review_input = st.text_input(label="Enter your movie review here:")
submit_button = st.button("Ask ML model:")

if submit_button:
    if review_input:
        result = generate_prediction(sa_model, wv_model, review_input)
        st.subheader("Sentiment is:")
        st.write(result)
    else:
        st.write("Enter a sentence first!")


# if __name__ == "__main__":
#     sa_model = load_sa_model()
#     test_sentence = "the movie was okay. I liked the apple scene"
#     result = generate_prediction(sa_model, test_sentence)
#     print(result)