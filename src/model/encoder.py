import os
import numpy as np
from PIL import Image
import torch
import clip
from transformers import CLIPProcessor, CLIPModel
import PIL
from datasets import Dataset, Image
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader

class Encoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained = True
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_images( self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
        def transform_fn(el):
            if isinstance(el['image'], PIL.Image.Image):
                imgs = el['image']
            else:
                imgs = [Image().decode_example(_) for _ in el['image']]
            return self.preprocess(images=imgs, return_tensors='pt')

        dataset = Dataset.from_dict({'image': images})
        dataset = dataset.cast_column('image',Image(decode=False)) if isinstance(images[0], str) else dataset
        dataset.set_format('torch')
        dataset.set_transform(transform_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        image_embeddings = []
        pbar = tqdm(total=len(images) // batch_size, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                image_embeddings.extend(self.model.get_image_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()
        return np.stack(image_embeddings)
    
    def encode_text( self, text: List[str], batch_size: int):
        dataset = Dataset.from_dict({'text': text})
        dataset = dataset.map(lambda el: self.preprocess(text=el['text'], return_tensors='pt', max_length=77, padding='max_length', truncation=True), batched=True, remove_columns=['text'])
        dataset.set_format('torch')
        dataloader = DataLoader(dataset, batch_size=batch_size)
        text_embeddings = []
        pbar = tqdm(total=len(text) // batch_size, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                text_embeddings.extend(self.model.get_text_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()
        return np.stack(text_embeddings)