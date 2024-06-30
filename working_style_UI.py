import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image






device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# original_img=load_image("cilian2.jpg")
# style_img=load_image("style.jpg")



#####################################################
# UI PART
import streamlit as st
import time

st.title("Photo Style Transfer app")
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
    for perc_completed in range(100):
        time.sleep(0.05)
        progress_bar.progress(perc_completed+1)
    col2.success("Photo uploaded successfully !!!")


uploaded_style_photo=col3.file_uploader("Upload your reference style....",on_change=change_style_photo_state)

if st.session_state["style_photo"]=="done":
    progress_bar=col3.progress(0)
    for perc_completed in range(100):
        time.sleep(0.05)
        progress_bar.progress(perc_completed+1)
    col3.success("Photo uploaded successfully !!!")
    style_img = load_image(uploaded_style_photo)

    original_img = load_image(uploaded_photo)
total_steps = st.slider('How Much Influential your Style Transfer should be', 400, 6000, 25)




    ##########################################################3


def main():
    model=models.vgg19(pretrained=True).features
    # ['0','5','10','19','28']
    # print(model)
    class VGG(nn.Module):
        def __init__(self):
            super(VGG,self).__init__()
            self.chosen_features=['0','5','10','19','28']
            self.model=models.vgg19(pretrained=True).features[:29]

        def forward(self,x):
            features=[]
            for layer_num,layer in enumerate(self.model):
                x=layer(x)
                if str(layer_num) in self.chosen_features:
                    features.append(x)

            return features


    model=VGG().to(device).eval()
    # generated=torch.randn(original_img.shape,device=device,requires_grad=True)
    generated=original_img.clone().requires_grad_(True)

    # Hyper parameters
    # total_steps=6000
    learning_rate=0.001
    alpha= 1
    beta = 0.01
    optimizer=optim.Adam([generated],lr=learning_rate)

    for step in range(total_steps):
        generated_features=model(generated)
        original_img_features=model(original_img)
        style_features=model(style_img)

        style_loss=original_loss= 0
        for gen_feature,orig_feature,style_feature in zip(generated_features,original_img_features,style_features):
            batch_size,channel,height,width = gen_feature.shape
            original_loss+=torch.mean((gen_feature-orig_feature)**2)

        #     COMPUTE GRAM MATRIX
            G=gen_feature.view(channel,height*width).mm(
                gen_feature.view(channel,height*width).t()
            )
            A= style_feature.view(channel,height*width).mm(
                style_feature.view(channel,height*width).t()
            )

            style_loss+=torch.mean((G-A)**2)

        total_loss=alpha*original_loss + beta + style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        st.toast("Please...PAY INR 5 To download your image")
        if step % 200 ==0:
            print(total_loss)
            save_image(generated,".\style\generated.png")
            st.image(Image.open(".\style\generated.png"), caption='Signup for more speedy generation and free credits..')




if __name__ =="__main__":
    st.button("Generate", on_click=main)
    # st.image(Image.open(".\style\generated.png"))

