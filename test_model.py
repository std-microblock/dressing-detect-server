from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_metric
import numpy as np
import torch
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor
from PIL import ImageDraw, ImageFont, Image, ImageFile
from datasets import load_dataset
import requests
ImageFile.LOAD_TRUNCATED_IMAGES = True
labels = ["crossdressing", "non-crossdressing"]

def predict(image):
    feature_extractor = ViTImageProcessor.from_pretrained(".\\vit-base-dress-detection\\")
    model = ViTForImageClassification.from_pretrained(".\\vit-base-dress-detection\\")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

import os
from PIL import Image

for filename in os.listdir('./tests'):
    path = os.path.join('./tests', filename)
    print(filename + ":")
    predict(Image.open(path))