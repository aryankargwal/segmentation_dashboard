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
    st.title("Segmentation Dashboard")
    st.text("A dashboard to check how various segmentation models perform with various types of images.")
      
    #widget to upload and display image
    upl = st.file_uploader('Upload the image you want to segment.')

    # displaying the image 
    if upl is not None:
        st.image(upl)
    
    #choosing the model to classify
    modelname1 = st.selectbox('Select First Segmentation Model',['DeepLabv3 ResNet 101','DeepLabv3 ResNet 50','DeepLabv3 MobileNetv3'])
    modelname2 = st.selectbox('Select Second Segmentation Model',['DeepLabv3 ResNet 101','DeepLabv3 ResNet 50','DeepLabv3 MobileNetv3'])
    
    #choosing the class that you want to be classified
    class_name = st.selectbox('Select Class that needs to be Classified',['__background__',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'])

    if st.button("Segment"):

        #transforming the image
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )]) 
        
        #loading the selected models
        # first model
        if modelname1 == 'DeepLabv3 ResNet 101':
            weights1, model1 = rn101()
        
        elif modelname1 == 'DeepLabv3 ResNet 50':
            weights1, model1 = rn50()
        
        elif modelname1 == 'DeepLabv3 ResNet 50':
            weights1, model1 = mnv3()

        # second model
        if modelname2 == 'DeepLabv3 ResNet 101':
            weights2, model2 = rn101()
        
        elif modelname2 == 'DeepLabv3 ResNet 50':
            weights2, model2 = rn50()
        
        elif modelname2 == 'DeepLabv3 ResNet 50':
            weights2, model2 = mnv3()

        input_image = Image.open(upl)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        col1, col2 = st.columns(2)

        # inference for first model
        with col1:
            st.subheader(modelname1)
            prediction = model1(input_batch)["out"]
            normalized_masks = prediction.softmax(dim=1)
            class_to_idx = {cls: idx for (idx, cls) in enumerate(weights1.meta["categories"])}
            mask1 = normalized_masks[0, class_to_idx[class_name]]
            output1 = to_pil_image(mask1)
            st.image(output1)

        #inference for second model
        with col2:
            st.subheader(modelname2)
            prediction = model2(input_batch)["out"]
            normalized_masks = prediction.softmax(dim=1)
            class_to_idx = {cls: idx for (idx, cls) in enumerate(weights2.meta["categories"])}
            mask2 = normalized_masks[0, class_to_idx[class_name]]
            output2 = to_pil_image(mask2)
            st.image(output2)
    
    
if __name__ == '__main__':
    main()