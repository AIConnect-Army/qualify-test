import torch
from torch import nn
import datasets
import glob, os, sys, pickle
from sklearn.model_selection import train_test_split
from modules.utils import load_yaml
from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation, 
    TrainingArguments, 
    Trainer
)
import numpy as np
import evaluate

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
PRETRAINED = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'


# Feature Extractor for Segformer
feature_extractor = SegformerFeatureExtractor.from_pretrained(PRETRAINED,size=384,reduce_labels=False)

def train_transforms(example_batch):
  images = [x for x in example_batch["image"]]
  labels = [x for x in example_batch["label"]]
  inputs = feature_extractor(images,labels,return_tensors="pt")
  return inputs

def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels,return_tensors="pt")
    return inputs

def compute_metrics(eval_pred, num_labels=4):
    metric = evaluate.load("mean_iou")

    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        print('중간실행')
        pred_labels = logits_tensor.detach().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        print('key,value 실행')
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        return metrics

if __name__ == "__main__":
    config_path = os.path.join(prj_dir, 'config', 'train.yaml')
    config = load_yaml(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE : {device}")


    # Load dataset from pickle files
    pickle_path = os.path.join(prj_dir, 'data', 'pickle', 'train.pkl')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            train_ds = pickle.load(f)
        with open(pickle_path.replace('train.pkl', 'valid.pkl'), 'rb') as f:
            val_ds = pickle.load(f)

    else:
        train_img_path = os.path.join(prj_dir, 'data', 'train', 'x', '*.png') 
        train_img_paths = glob.glob(train_img_path)

        #train_img_paths, val_img_paths = train_test_split(train_img_paths, test_size=config['val_size'], random_state=config['seed'], shuffle=True)

        train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/new_data/train/x/r0_x/*.png')
        train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/new_data/train/x/r1_x/*.png')
        #train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/new_data/train/x/r2_x/*.png')
        #train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/new_data/train/x/r3_x/*.png')

        #train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/add_cut/train/x/r2_x/*.png')
        #train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/add_cut/train/x/r3_x/*.png')
        #train_img_paths += glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/add_cut/train/x/r1_x/*.png')
        #train_img_paths += glob.glob(train_img_path.replace('train', 'newcut'))

        train_img_paths, val_img_paths = train_test_split(train_img_paths, test_size=config['val_size'],
                                                          random_state=config['seed'], shuffle=True)
        train_img_paths += glob.glob(train_img_path.replace('train', 'up'))
        train_img_paths += glob.glob(train_img_path.replace('train', 'cut'))
        rotate_cut = glob.glob('D:/qualify-test-main (5)/qualify-test-main/data/add_cut/train/x/r1_x/*.png')
        train_img_paths += rotate_cut[:800]
        train_ds = datasets.Dataset.from_dict(
            {"image": train_img_paths, "label": list(map(lambda x : x.replace('x', 'y'),train_img_paths))}, 
            features=datasets.Features({"image": datasets.Image(), "label": datasets.Image()}),
            split="train")

        val_ds = datasets.Dataset.from_dict(
            {"image": val_img_paths, "label": list(map(lambda x : x.replace('x', 'y'),val_img_paths))}, 
            features=datasets.Features({"image": datasets.Image(), "label": datasets.Image()}),
            split="val")
        train_ds.shuffle(seed=42)
        os.makedirs(os.path.join(prj_dir, 'data','pickle'), exist_ok=True)

        with open(pickle_path, 'wb') as f:
            pickle.dump(train_ds, f)
        with open(pickle_path.replace('train', 'valid'), 'wb') as f:
            pickle.dump(val_ds, f)

        print(f"PICKLE SAVED")

    print("=====DATASET LOADED=====")
    print(f"TRAIN : {train_ds}, VAL : {val_ds}")

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)

    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED,num_labels=config['n_classes'],ignore_mismatched_sizes=True).to(device)

    output_path = os.path.join(prj_dir, 'results', 'train', config['train_folder'])
    training_args = TrainingArguments(
        output_dir = output_path,
        learning_rate=0.00005,
        num_train_epochs=config['n_epochs'],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=4486,
        eval_steps=4486,
        load_best_model_at_end = True,
        logging_steps=500,
        eval_accumulation_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers = 1,
        warmup_ratio = 0.00001,
        warmup_steps=1121,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
