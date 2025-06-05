import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image
import os

# Load model and tokenizer
@st.cache_resource
def load_caption_model():
    model = load_model("/Users/prarabdhapandey/caption_app/image_caption_model.h5")
    with open("/Users/prarabdhapandey/caption_app/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Feature extractor
def extract_features(img_path):
    model = ResNet50(include_top=False, pooling='avg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x, verbose=0)

# Caption generator
def generate_caption(model, tokenizer, photo, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Streamlit UI
st.title("🖼️ AI Image Caption Generator")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    img = Image.open("temp.jpg")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        model, tokenizer = load_caption_model()
        photo = extract_features("temp.jpg")
        caption = generate_caption(model, tokenizer, photo)
    st.success("Caption Generated:")
    st.markdown(f"### 📝 {caption}")
