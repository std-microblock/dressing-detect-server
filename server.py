from flask import Flask, request
from PIL import Image
import io
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor
import numpy as np

app = Flask(__name__)

feature_extractor = ViTImageProcessor.from_pretrained(".\\vit-base-dress-detection\\")
model = ViTForImageClassification.from_pretrained(".\\vit-base-dress-detection\\")

def predict(image):
    

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


@app.route("/predict", methods=["POST"])
def route():
    image = request.files.get("image").read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = np.array(image)
    result = predict(image)
    
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0")