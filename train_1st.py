from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_metric
import numpy as np
import torch
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor
from PIL import ImageDraw, ImageFont, Image, ImageFile
from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


labels = ["crossdressing", "non-crossdressing"]

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def transform(batch):
    batchTs = [x.convert("RGB") for x in batch['image']]
    inputs = feature_extractor(
        images=batchTs, return_tensors='pt', padding=True)

    inputs['labels'] = batch['label']
    return inputs


ds = load_dataset(
    "imagefolder", data_dir="./dataset").with_transform(transform)

ds_val = load_dataset(
    "imagefolder", data_dir="./dataset_validation").with_transform(transform)


training_args = TrainingArguments(
    output_dir="./vit-base-dress-detection",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)


print(ds['train'])

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=ds['train'],
    eval_dataset=ds_val['train'],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(ds_val['train'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
