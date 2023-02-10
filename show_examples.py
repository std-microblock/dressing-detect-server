import random
from PIL import ImageDraw, ImageFont, Image, ImageFile
from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset = load_dataset("imagefolder", data_dir="./dataset_validation")

def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):
    w, h = size
    labels = ["crossdressing","non-crossdressing"]
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("J:\\Oculus\\Support\\oculus-dash\\dash\\data\\fonts\\NotoSansCJKtc-Bold.ttf", 24)

    for label_id, label in enumerate(labels):
        ds_slice = ds['train'].filter(lambda ex: ex['label'] == label_id).shuffle(seed).select(range(examples_per_class))

        for i, example in enumerate(ds_slice):
            image = example['image']
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255), font=font)

    return grid

show_examples(dataset, seed=random.randint(0, 1337), examples_per_class=3).save("./dataset_examples.png")
