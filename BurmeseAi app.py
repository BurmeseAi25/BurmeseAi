import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("model.h5")

def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert("L")  # grayscale
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    result = model.predict(img)
    return np.argmax(result)

def main():
    st.title("မြန်မာ AI - လက်ဖြင့်ရေးထိုးစာလုံးခန့်မှန်းခြင်း")

    uploaded_file = st.file_uploader("ပုံဖိုင်တင်ပါ", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="တင်ထားသောပုံ", use_column_width=True)
        prediction = predict_digit(img)
        st.success(f"ခန့်မှန်းထားသောနံပါတ်မှာ - {prediction}")

if __name__ == "__main__":
    main()
