from PIL import Image
import os
import random
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
    feature_extractor = ViTImageProcessor.from_pretrained(
        ".\\vit-base-dress-detection\\")
    model = ViTForImageClassification.from_pretrained(
        ".\\vit-base-dress-detection\\")

    inputs = feature_extractor(images=image.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


ImageFile.LOAD_TRUNCATED_IMAGES = True


xplot = 3
w, h = (224, 224)
labels = ["crossdressing", "non-crossdressing"]

tests=os.listdir('./tests')
grid = Image.new('RGB', size=(
    xplot * w, len(tests) // xplot * h))
draw = ImageDraw.Draw(grid)
font = ImageFont.truetype(
    "J:\\Oculus\\Support\\oculus-dash\\dash\\data\\fonts\\NotoSansCJKtc-Bold.ttf", 24)


for i, filename in enumerate(tests):
    try:
        path = os.path.join('./tests', filename)
        image = Image.open(path)
        idx = i
        box = (idx % xplot * w, idx // xplot * h)
        grid.paste(image.resize((w, h)), box=box)
        draw.text(box, predict(image.convert("RGB")), (255, 255, 255),
                font=font, stroke_width=2, stroke_fill="black")
        print("{}/{}    \r",i,len(tests))
    except:
        print("Failed {filename}")

grid.save("./model_test.png")
