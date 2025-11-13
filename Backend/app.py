from flask import Flask, request, jsonify, send_from_directory
import torch as t
import torch.nn as nn
import numpy as np
from flask_cors import CORS
import torchvision.transforms as transforms
import base64
import io
import cv2
from PIL import Image


app = Flask(__name__, static_folder="../Frontend/dist", static_url_path="")

CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})


import torch
import torch.nn as nn
import torch.nn.functional as F
def conv(input_channels,output_channels):
    return  nn.Sequential(
        nn.Conv2d(input_channels,output_channels,3,1,1),
        nn.ReLU(),
        nn.Conv2d(output_channels,output_channels,3,1,1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.layer = nn.Sequential(
            conv(3,64),
            conv(64,128),
            conv(128,256),
            conv(256,512),
            conv(512,512),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(18432, 4096),  
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(4096, 10)
        )

    def forward(self,x):
        x = self.layer(x)
        x = t.flatten(x, 1) 
        x = self.fc(x)
        return x


model = VGG16()  
model.load_state_dict(t.load("model.pth"))
model.eval()  


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_base64 = data['image']

    
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(image.size)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    test_img = transform(image)
    test_img = test_img.unsqueeze(0)  
    


    lab= ['A', 'B', 'C', 'L', 'Nothing', 'O', 'V', 'W', 'Z']
    
    with t.no_grad():
        out = model(test_img)
        probs = nn.Softmax(dim=1)(out) 
        _, predicted = t.max(out.data, 1)

        print(t.argmax(probs, dim=1).item())
        print(predicted)
    return jsonify({
    "status": "success",
    "prediction": lab[(predicted.item())-1],
    "probability": 0
})



if __name__ == "__main__":
    app.run(debug=True)
