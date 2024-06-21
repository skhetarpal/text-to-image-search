import os
import zipfile
import requests
from tqdm import tqdm
import json
import shutil
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision import transforms as torchvision_transforms
import json
from torch import nn
from torch.utils.data import DataLoader, random_split
if not importlib.util.find_spec('transformers'):
  !pip install transformers
from transformers import BertTokenizer, BertModel
from torchvision.models import inception_v3, Inception_V3_Weights
import h5py
from itertools import chain
import pickle
from torch import matmul
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from data_preprocessing import *



class image_and_captions_dateset(torch.utils.data.Dataset):
    """
    Class to provide a custom dataset for the images and captions

    Attributes
    ----------
    annotations_file : str
        The path to the annotations file
    img_dir : str
        The path to the directory containing the images
    image_transform : torchvision.transforms.Compose
        The transform to apply to the images

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset
    __getitem__(idx)
        Returns the transformed image, the caption, and the sample Id at index idx
    """
    def __init__(self, annotations_file, img_dir, image_transform):

        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        self.img_dir = img_dir
        self.image_transform = image_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample_id = self.annotations[idx]['image_id']
        img_path = os.path.join(self.img_dir, 'COCO_train2014_%012d.jpg'%sample_id)
        image = read_image(img_path)
        if image.shape[0] == 1: image = image.tile((3,1,1))
        caption = self.annotations[idx]['caption']
        image = self.image_transform(image)
        return image, caption, sample_id


class collate_fn():
  """
  Class to provide a custom collate function that modifies the batching process.

  Attributes
  ----------
  tokenizer : BertTokenizer
      The tokenizer to use for tokenizing the captions.

  Methods
  -------
  __call__(batch)
      Takes in a batch of captions and returns a batch of tokenized captions
      padded to the max caption length within the batch.
  """
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
  def __call__(self, batch):
    [images, captions, sample_ids] = list(zip(*batch))
    embedded_captions = self.tokenizer(list(captions), padding='longest', truncation=True, return_tensors='pt')
    images = torch.stack(list(images))
    return (images, embedded_captions, captions, sample_ids)


class projection_head(nn.Module):
    """
    Module to provide the projection head that will be used to project the outputs of both the text and image models into the same embedding space.

    Attributes
    ----------
    in_features : int
        The number of features outputted by the base model.
    projection_dims : int
        The number of features in the text/image common embedding space.
    dropout_rate : float
        The dropout rate to use
    """
    def __init__(self, in_features, projection_dims, dropout_rate):
        super().__init__()
        self.lin1 = nn.Linear(in_features, projection_dims)
        self.gelu = nn.GELU()
        self.layernorm1 = nn.LayerNorm(projection_dims)
        self.lin2 = nn.Linear(projection_dims, projection_dims)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layernorm2 = nn.LayerNorm(projection_dims)
    def forward(self, x):
        x1 = self.lin1(x)
        x = self.gelu(x1)
        x = self.layernorm1(x)
        x = self.lin2(x)
        x = self.dropout(x)
        x = self.layernorm2(x)
        return x
    

class Embeddings_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_embeddings, caption_embeddings, caption, sample_id):

        self.image_embeddings = image_embeddings
        self.caption_embeddings = caption_embeddings
        self.caption = caption
        self.sample_id = sample_id

    def __len__(self):
        return len(self.caption_embeddings)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.caption_embeddings[idx], self.caption[idx], self.sample_id[idx]


class custom_loss(nn.Module):
    def forward(self, image_out, caption_out):
        predicted_sim = matmul(image_out, torch.transpose(caption_out, 0, 1))
        image_sim = matmul(image_out, torch.transpose(image_out, 0, 1))
        caption_sim = matmul(caption_out, torch.transpose(caption_out, 0, 1))
        targets = F.softmax((image_sim+caption_sim)/2, dim=0)
        return (F.cross_entropy(predicted_sim, targets) + F.cross_entropy(torch.transpose(predicted_sim, 0, 1), targets)) / 2


def train_projection_head_one_epoch(image_embedder, text_embedder, dataloader, loss_fn, optimizer):
    image_embedder.train(); text_embedder.train()
    for batch, (images, captions, _, _) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        images_out = image_embedder(images)
        captions_out = text_embedder(captions)
        loss = loss_fn(images_out, captions_out)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def validate_projection_head(image_embedder, text_embedder, dataloader, loss_fn):
    image_embedder.eval(); text_embedder.eval()
    with torch.no_grad():
        val_loss = 0
        for batch, (images, captions, _, _) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            images_out = image_embedder(images)
            captions_out = text_embedder(captions)
            val_loss += loss_fn(images_out, captions_out)
        return val_loss / len(dataloader)

def train_projection_head(image_embedder, text_embedder, train_dataloader, val_dataloader, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        # print('Epoch', epoch)
        val_loss = validate_projection_head(image_embedder, text_embedder, val_dataloader, loss_fn)
        # print('Val_loss:', val_loss)
        train_projection_head_one_epoch(image_embedder, text_embedder, train_dataloader, loss_fn, optimizer)
        scheduler.step(val_loss)

    return image_embedder, text_embedder

def train_full_model_one_epoch(image_embedder, text_embedder, dataloader, loss_fn, optimizer):
    image_embedder.train(); text_embedder.train()
    for batch, (images, captions, _, _) in enumerate(tqdm(dataloader, desc='Epoch Progress')):
        images = images.to(device)
        for item in captions:
          captions[item] = captions[item].to(device)
        images_out = image_embedder(images)
        captions_out = text_embedder(captions)
        loss1 = loss_fn(images_out.logits, captions_out)
        loss2 = loss_fn(images_out.aux_logits, captions_out)
        loss = loss1 + 0.4 * loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def validate_full_model(image_embedder, text_embedder, dataloader, loss_fn):
    image_embedder.eval(); text_embedder.eval()
    with torch.no_grad():
        val_loss = 0
        for batch, (images, captions, _, _) in enumerate(tqdm(dataloader, desc='Calculating Val Loss')):
            images = images.to(device)
            for item in captions:
              captions[item] = captions[item].to(device)
            images_out = image_embedder(images)
            captions_out = text_embedder(captions)
            val_loss += loss_fn(images_out, captions_out)
        return val_loss / len(dataloader)

def train_full_model(image_embedder, text_embedder, train_dataloader, val_dataloader, epochs, loss_fn, optimizer):
    for epoch in range(epochs):
        print('Epoch', epoch)
        val_loss = validate_full_model(image_embedder, text_embedder, val_dataloader, loss_fn)
        print('Val_loss:', val_loss)
        train_full_model_one_epoch(image_embedder, text_embedder, train_dataloader, loss_fn, optimizer)
        scheduler.step(val_loss)
    val_loss = validate_full_model(image_embedder, text_embedder, val_dataloader, loss_fn)
    print('Val_loss:', val_loss)

    return image_embedder, text_embedder



def performance_eval(image_proj_embeddings, caption_proj_embeddings, num_to_print, img_dir):

    # Array to store the image that most closely aligns with each caption's projection
    selected_images = np.empty(len(caption_proj_embeddings)).astype('int')

    k = round(len(image_proj_embeddings)*.05)
    counter = 0
    for i, caption in enumerate(caption_proj_embeddings):
      predicted_sim = matmul(caption, torch.transpose(image_proj_embeddings, 0, 1))
      selected_images[i] = torch.argmax(predicted_sim)
      if i in torch.topk(predicted_sim, k=k)[1]:
        counter += 1

    print('Accuracy of', counter/len(caption_proj_embeddings))

    # Print some captions and corresponding chosen images
    for i in range(num_to_print):
        print('Caption:', captions_list[i])
        print('Correct Image:', sample_ids_list[i])
        img_path = os.path.join(img_dir, 'COCO_train2014_%012d.jpg'%sample_ids_list[i])
        image = read_image(img_path)
        plt.imshow(image.permute(1, 2, 0)); plt.show()
        print('Selected Image:', sample_ids_list[selected_images[i]])
        selected_image_id = sample_ids_list[selected_images[i]]
        img_path = os.path.join(img_dir, 'COCO_train2014_%012d.jpg'%selected_image_id)
        image = read_image(img_path)
        plt.imshow(image.permute(1, 2, 0)); plt.show()



if __name__ == "__main__":
    
    # Generate the PyTorch dataset
    annotations_file = "./data/annotations/captions_train2014.json"
    img_dir = './data/train2014'
    coco_dataset = image_and_captions_dateset(annotations_file, img_dir, image_transform())
    
    
    # Split the dataset.  We won't use the whole dataset because it is too large
    splits = (0.4, 0.1, 0.1)
    train_size = int(splits[0] * len(coco_dataset))
    val_size = int(splits[1] * len(coco_dataset))
    test_size = int(splits[2] * len(coco_dataset))
    extra = len(coco_dataset) - train_size - val_size - test_size
    train_dataset, val_dataset, test_dataset, _ = random_split(coco_dataset, [train_size, val_size, test_size, extra])
    
    # Create DataLoader for each dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataloader = DataLoader(train_dataset, batch_size = 16, collate_fn=collate_fn(tokenizer), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = 16, collate_fn=collate_fn(tokenizer), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 16, collate_fn=collate_fn(tokenizer), shuffle=False)
    
    # Load in the BERT model and the Inception model, and create projection heads for each
    image_embedder = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    image_projection_head = projection_head(in_features=image_embedder.fc.in_features, projection_dims=256, dropout_rate=0.1)
    image_AuxLogits_projection_head = projection_head(in_features=image_embedder.AuxLogits.fc.in_features, projection_dims=256, dropout_rate=0.1)
    image_embedder.fc = torch.nn.Identity()
    image_embedder.AuxLogits.fc = torch.nn.Identity()
    caption_embedder = BertModel.from_pretrained('bert-base-uncased')
    caption_projection_head = projection_head(in_features=caption_embedder.pooler.dense.out_features, projection_dims=256, dropout_rate=0.1)
    
    # Move all elements of the dual encoder to the GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_embedder.to(device)
    caption_embedder.to(device)
    image_projection_head.to(device)
    caption_projection_head.to(device)
    
    # Establish the target location at which to save the base-embeddings
    base_embeddings_file = 'data/base_embeddings.h5'
    base_model_embeddings_metadata_file = "data/base_embeddings_metadata.pkl"
    
    # Generate the base-embeddings
    if not os.path.isfile(base_embeddings_file):
        generate_base_embeddings(image_embedder, caption_embedder, train_dataloader, base_embeddings_file, base_model_embeddings_metadata_file)
    
    # Load the base-embeddings, as well as the captions and sample IDs
    data = h5py.File(base_embeddings_file, 'r')
    image_embeddings = torch.tensor(data['image_embeddings'], dtype=torch.float)
    caption_embeddings = torch.tensor(data['caption_embeddings'], dtype=torch.float)
    
    with open(base_model_embeddings_metadata_file, "rb") as file:
        (caption_list, sample_id_list) = pickle.load(file)
    
    # Generate the datasets for the pre-embedded data
    validation_cutoff = round(len(image_embeddings)*0.05)
    embeddings_train_dataset = Embeddings_Dataset(image_embeddings[:-validation_cutoff], caption_embeddings[:-validation_cutoff], caption_list[:-validation_cutoff], sample_id_list[:-validation_cutoff])
    embeddings_train_dataloader = DataLoader(embeddings_train_dataset, batch_size=16, shuffle=True)
    embeddings_val_dataset = Embeddings_Dataset(image_embeddings[-validation_cutoff:], caption_embeddings[-validation_cutoff:], caption_list[-validation_cutoff:], sample_id_list[-validation_cutoff:])
    embeddings_val_dataloader = DataLoader(embeddings_val_dataset, batch_size=16, shuffle=True)
    
    
    # Train the projection heads

    image_projection_head.to(device)
    caption_projection_head.to(device)
    
    loss_fn = custom_loss()
    optimizer = torch.optim.AdamW(chain(image_projection_head.parameters(), caption_projection_head.parameters()), lr=0.2, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=2e-2, step_size_up=200, mode='triangular')
    
    image_projection_head, caption_projection_head = train_projection_head(image_projection_head,
                                                                           caption_projection_head,
                                                                           embeddings_train_dataloader,
                                                                           embeddings_val_dataloader,
                                                                           100,
                                                                           loss_fn,
                                                                           optimizer)
    
    os.makedirs('/content/drive/My Drive/Projects/text-to-image-search/models', exist_ok=True)
    torch.save(caption_projection_head, '/content/drive/My Drive/Projects/text-to-image-search/models/caption_projection_head.pt')
    torch.save(image_projection_head, '/content/drive/My Drive/Projects/text-to-image-search/models/image_projection_head.pt')
    print('Saved!')
    
    
    # Evaluate the performance of the projection heads
    
    image_proj_embeddings, caption_proj_embeddings, captions_list, sample_ids_list = \
        generate_projections(image_projection_head, caption_projection_head, embeddings_val_dataloader)
    
    performance_eval(image_proj_embeddings, caption_proj_embeddings, 5, img_dir)
    
    
    # Create the image side of the dual encoder.  It will still be called the image_embedder.

    image_embedder.fc = image_projection_head
    image_embedder.AuxLogits.fc = image_AuxLogits_projection_head
    
    
    # Create the text side of the dual encoder.  We will call this the text_embedder
    
    class CaptionEmbedderWithProjectionHead(nn.Module):
      def __init__(self, caption_embedder, caption_projection_head):
        super().__init__()
        self.base_model = caption_embedder
        self.projection_head = caption_projection_head
    
      def forward(self, x):
        base_model_output = self.base_model(**x)
        return self.projection_head(base_model_output.pooler_output)
    
    text_embedder = CaptionEmbedderWithProjectionHead(caption_embedder=caption_embedder, caption_projection_head=caption_projection_head)
    
    
    # Train the full dual encoder

    image_embedder.to(device)
    text_embedder.to(device)
    
    loss_fn = custom_loss()
    optimizer = torch.optim.AdamW(chain(image_embedder.parameters(), text_embedder.parameters()), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    image_embedder, text_embedder = train_full_model(image_embedder, text_embedder, train_dataloader, val_dataloader, 1, loss_fn, optimizer)
    
    torch.save(image_embedder, '/content/drive/My Drive/Projects/text-to-image-search/models/image_embedder.pt')
    torch.save(text_embedder, '/content/drive/My Drive/Projects/text-to-image-search/models/text_embedder.pt')
    print('Saved!')
    
    # Evaluate the performance of the dual encoder
    
    image_proj_embeddings, caption_proj_embeddings, captions_list, sample_ids_list = \
        generate_projections(image_embedder, text_embedder, test_dataloader)
    
    performance_eval(image_proj_embeddings, caption_proj_embeddings, 5, img_dir)