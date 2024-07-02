import streamlit as st
import time
import subprocess
from PIL import Image


device="cuda"

import torchvision.transforms as transforms
image_size=356
loader=transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[],std=[])
    ]
)
def load_image(image_name):
    image=Image.open(image_name)
    image=loader(image).unsqueeze(0)
    return image.to(device)

# subprocess.run(["python", "script.py"])

st.title("Photo Style Transfer App")
col1,col2,col3=st.columns([1,3,3])

if "photo" not in st.session_state:
    st.session_state["photo"]="not done"
if "style_photo" not in st.session_state:
    st.session_state["style_photo"]="not done"
def change_photo_state():
    st.session_state["photo"]="done"
def change_style_photo_state():
    st.session_state["style_photo"]="done"


st.markdown("**Either You can upload a photo or take a picture _now..._**.")

uploaded_photo=col2.file_uploader("Upload your photo to be transfered..",on_change=change_photo_state)
camera_photo=col2.camera_input("Take a photo..",on_change=change_photo_state)

if st.session_state["photo"]=="done":

    progress_bar=col2.progress(0)
    # for perc_completed in range(100):
    #     time.sleep(0.05)
    #     progress_bar.progress(perc_completed+1)
    col2.success("Photo uploaded successfully !!!")
    original_img = load_image(uploaded_photo)


uploaded_style_photo=col3.file_uploader("Upload your reference style....",on_change=change_style_photo_state)

if st.session_state["style_photo"]=="done":
    progress_bar=col3.progress(0)
    # for perc_completed in range(100):
    #     time.sleep(0.05)
    #     progress_bar.progress(perc_completed+1)
    col3.success("Photo uploaded successfully !!!")

total_steps = st.slider('How Much Influential your Style Transfer should be', 400, 6000, 25)


# image = Image.open("C:\\Users\\shitosu\\Desktop\\prog\\GANN\\style\\cilian.jpg")

# st.image(image, caption='Sunrise by the mountains')