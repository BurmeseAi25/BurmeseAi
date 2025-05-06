import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("model.h5")

def predict(img):
    img = img.resize((28, 28))
    img = np.array(img.convert("L"))
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction)

st.title("Handwritten Digit Recognition (Myanmar)")
uploaded_image = st.file_uploader("ဓာတ်ပုံတင်ပါ", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="တင်ထားသောဓာတ်ပုံ", use_column_width=True)
    st.write("ခနနဲ့ခနပေါ်မယ်...")
    result = predict(img)
    st.success(f"ခန့်မှန်းထားသောနံပါတ်: {result}")
