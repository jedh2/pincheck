import os
import torch
import boto3
from torchvision import transforms
from PIL import Image
from app.gradcam import generate_gradcam
import urllib.request

S3_URLS = {
    'front': 'https://pincheck-models.s3.us-east-2.amazonaws.com/model_front.pth',
    'back': 'https://pincheck-models.s3.us-east-2.amazonaws.com/model_back.pth'
}

model_paths = {
    'front': 'app/model_front.pth',
    'back': 'app/model_back.pth'
}

s3_bucket = 'pincheck-models'
s3_keys = {
    'front': 'model_front.pth',
    'back': 'model_back.pth'
}

def download_model_if_needed(side):
    local_path = model_paths[side]
    if not os.path.exists(local_path):
        print(f"Downloading {side} model from S3...")
        s3 = boto3.client('s3')
        s3.download_file(s3_bucket, s3_keys[side], local_path)

def load_model(side):
    download_model_if_needed(side)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_paths[side], map_location='cpu'))
    model.eval()
    return model

def predict_image(file, side):
    model = load_model(side)
    image = Image.open(file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()
    label = 'real' if pred == 1 else 'fake'

    gradcam_path = generate_gradcam(model, input_tensor, side)

    return label, gradcam_path