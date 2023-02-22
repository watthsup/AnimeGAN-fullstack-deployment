import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

def convert_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    if option == 'face anime V1':
        model_name = "face_anime_v1"
    elif option == 'face anime V2 + Beutify':
        model_name = "face_anime_v2"
    image = Image.open(upload)
    with st.spinner('Hold on tight!!!'):
        st.write("Original image :camera:")
        st.image(image)
        response = requests.post(f"http://backend:8000/models/infer?model_name={model_name}", files={"file": upload.getvalue()})
    st.balloons()
    st.success('Done')
    content = response.content
    concat_img = Image.open(io.BytesIO(content))
    concat_array = np.array(concat_img)
    face_array = concat_array[:,:512,:]
    anime_array= concat_array[:,512:,:]
    
    face_img = Image.fromarray(np.uint8(face_array)).convert('RGB')
    col1.write("Detected face image :camera:")
    col1.image(face_img)

    anime_img = Image.fromarray(np.uint8(anime_array)).convert('RGB')
    col2.write("Anime face image :grin:")
    col2.image(anime_img)

    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(anime_img), "fixed.png", "image/png")


st.set_page_config(layout="wide", page_title="Anime Style Transfer APP")

st.write("## Turn your portrait photo into cartoon style")
st.write(
    ":grin: Upload your selfies, and let AI turn yourself in to cartoon character. Full quality images can be downloaded from the sidebar. :grin:"
)
st.sidebar.write("## Upload and download :gear:")
option = st.sidebar.selectbox(
        "Model Selection",
        ("", "face anime V1", "face anime V2 + Beutify"),
    )

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None: 
    if option != "":
        fix_image(upload=my_upload)
    else:
        st.warning('Please select the model', icon="⚠️")

