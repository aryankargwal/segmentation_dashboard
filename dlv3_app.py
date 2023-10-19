#imports
import streamlit as st
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import torch
from PIL import Image

#command to stop file encoder warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# Some utils to make things faster 
@st.cache_resource()
def read_classes():
    
    with open('classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return classes

#storing models in cache
@st.cache_resource
def rn101():
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    return weights, model

def rn50():
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()
    return weights, model

def mnv3():
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()
    return weights, model

#function to show image
def showimg(x):
    st.image(x, use_column_width=False)
    
#main function
def main():

    upl = None 

    #title and header
    st.title("Classification Dashboard")
    st.text("A dashboard to check how various segmentation models perform with various types of images.")
      
    #widget to upload and display image
    upl = st.file_uploader('Upload the image you want to segment.')

    # displaying the image 
    if upl is not None:
        st.image(upl)
    
    #choosing the model to classify
    modelname = st.selectbox('Select Segmentation Model',['DeepLabv3 ResNet 101','DeepLabv3 ResNet 50','DeepLabv3 MobileNetv3'])
    
    if st.button("Classify"):

        #transforming the image
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )]) 
        
        #loading the selected model
        if modelname == 'DeepLabv3 ResNet 101':
            weights, model = rn101()
        
        elif modelname == 'DeepLabv3 ResNet 50':
            weights, model = rn50()
        
        elif modelname == 'DeepLabv3 ResNet 50':
            weights, model = mnv3()

        input_image = Image.open(upl)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        prediction = model(input_batch)["out"]
        normalized_masks = prediction.softmax(dim=1)
        class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        mask = normalized_masks[0, class_to_idx["car"]]
        output = to_pil_image(mask)
        st.image(output)

        #loading ImageNet classes
        classes = read_classes()
    
    
if __name__ == '__main__':
    main()