from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

settings = [
    {"do_resize": True, "do_normalize": True, "return_tensors": None},
    {"do_resize": True, "do_normalize": True, "return_tensors": "pt"},
    {"do_resize": False, "do_normalize": True, "return_tensors": None},
    {"do_resize": False, "do_normalize": True, "return_tensors": "pt"},
    {"do_resize": True, "do_normalize": False, "return_tensors": None},
    {"do_resize": True, "do_normalize": False, "return_tensors": "pt"},
    {"do_resize": False, "do_normalize": False, "return_tensors": None},
    {"do_resize": False, "do_normalize": False, "return_tensors": "pt"},
]
for kwargs in settings:
    input_single = feature_extractor(images=image, **kwargs)['pixel_values']
    input_batch = feature_extractor(images=[image, image], **kwargs)['pixel_values']
    print("\n" + str(kwargs))
    print(f"Single image    - type: {type(input_single)}, shape: {[x.shape for x in input_single] if isinstance(input_single, list) else input_single.shape}")
    print(f"Batch of images - type: {type(input_batch)}, shape: {[x.shape for x in input_batch] if isinstance(input_batch, list) else input_batch.shape}")