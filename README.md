# Semantic Segmentation Dashboard
Repository to hold, test and compare state-of-the-art semantic segmentation models. In this repository we are going to be comparing these models on different parameters and using different datasets. The main motive of this repository is to finalize on a segmentation model to use for our Ice Segmentation Task.

## Problem Statement
We are trying to deploy a web application running on streamlit to explore different versions of DeepLab v3 Architecture and more such segmentation architectures to explore the performance and live inference of the models.

## Segmentation Models
What are some lightweight, high Mean IoU Models that can be added to the dashboard:

- [x] Deeplabv3 MobileNetv3
- [x] Deeplabv3 ResNet50
- [x] Deeplabv3 ResNet101
- [ ] Deeplabv3+
- [ ] Deeplabv3 JFT
- [ ] PSPNet
- [ ] EncNet
- [ ] Discriminative Feature Network

## Steps of Deployment

- [ ] Individual Notebooks for Model Inference
- [ ] Make app for dashboard using streamlit
- [ ] Add more models
- [ ] Stats for Inference

## Run it Yourself
- Cloning the Repository: 

        git clone https://github.com/aryankargwal/semantic_segmentation.git

- Setting up the Python Environment with dependencies 

        pip install -r requirements.txt

- Run Streamlit App
    ```
    streamlit run app.py
    ```
- Run Inference using the Application

<img src = "tutorial/inference.gif">

## License
This repo is under the MIT License. See [License](License) for details.
