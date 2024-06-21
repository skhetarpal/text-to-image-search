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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def download_and_extract(url):
        # Function that downloads and extracts files given a url

        os.makedirs('data', exist_ok=True)

        # Extract the filename from the URL
        filename = os.path.join('data', url.split("/")[-1])

        # Download the file if it doesn't exist
        if not os.path.exists(filename):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

        # Extract the zip file
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data')
            print(f"Extracted all contents to data")


def download_ms_coco():
    # Function to download the MS-COCO dataset (2017 version)
    
    URLS = {
        "train_images": "http://images.cocodataset.org/zips/train2014.zip",
        "val_images": "http://images.cocodataset.org/zips/val2014.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    }

    # Download, extract images and annotations
    for url in URLS.values():
        download_and_extract(url)

    print("MS-COCO dataset downloaded and extracted successfully!")


class resize_and_pad_image:
    # Class to ensure that all images are the same size
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        h, w = image.size()[-2:]
        size_param = int(min(w/h, h/w) * self.size)
        image = torchvision_transforms.functional.resize(img=image, size=size_param)
        h_new, w_new = image.size()[-2:]
        dw = (self.size-w_new)//2
        dh = (self.size-h_new)//2
        rw = (self.size-w_new)%2
        rh = (self.size-h_new)%2
        return torchvision_transforms.functional.pad(image, padding=(dw, dh, dw+rw, dh+rh))


def image_transform():
    return torchvision_transforms.Compose(
        [resize_and_pad_image(size=299),
         torchvision_transforms.ConvertImageDtype(torch.float32),
         torchvision_transforms.Lambda(lambda x: x / 255),
         torchvision_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )


def generate_base_embeddings(image_embedder, text_embedder, dataloader, base_embeddings_file, base_model_embeddings_metadata_file):
    """
    Function to generate and save the base embeddings for the entire dataset.
    The embeddings are collected in two lists and then saved into an HDF5 file and a pickle file.

    Inputs
    -------
    image_embedder : torch.nn.Module
        The image model to use for embedding the images.
    text_embedder : torch.nn.Module
        The text model to use for embedding the captions.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for generating the original dataset's data.
    base_embeddings_file : str
        The path to the file to store the base embeddings.
    base_model_embeddings_metadata_file : str
        The path to the file to store the original captions and sample ids.

    Outputs
    -------
    h5py file is stored at base_embeddings_file
    """

    image_embedder.eval(); text_embedder.eval()

    with torch.no_grad():

        # Create a temporary file to store the embeddings
        with h5py.File('temporary_file.h5', 'w') as h5f:

            # Create a dataset within the HDF5 file to store embeddings
            image_embeddings_h5 = h5f.create_dataset("image_embeddings", (len(dataloader.dataset), 2048), dtype='float32')
            caption_embeddings_h5 = h5f.create_dataset("caption_embeddings", (len(dataloader.dataset), 768), dtype='float32')

            # Lists to store the data as it is embedded in batches
            caption_list = []
            sample_id_list = []

            for batch, (images, tokenized_captions, captions, sample_ids) in enumerate(tqdm(dataloader, desc='Embedding Data')):
                images = images.to(device)
                for item in tokenized_captions:
                  tokenized_captions[item] = tokenized_captions[item].to(device)
                images_out = image_embedder(images)
                captions_out = caption_embedder(**tokenized_captions).pooler_output
                image_embeddings_h5[batch*16:(batch*16+len(images_out))] = images_out.cpu().numpy()
                caption_embeddings_h5[batch*16:(batch*16+len(images_out))] = captions_out.cpu().numpy()
                caption_list = caption_list + list(captions)
                sample_id_list = sample_id_list + list(sample_ids)

    !mv temporary_file.h5 {base_embeddings_file}
    
    with open('temporary_file.pkl', "wb") as file:
        pickle.dump((caption_list, sample_id_list), file)
    !mv temporary_file.pkl {base_model_embeddings_metadata_file}


def generate_projections(image_model, caption_model, dataloader):
    """
    Function to generate (but not save) the projects for an entire dataset.
    The projections are collected in two large torch.Tensors, and projection metadata is collected into two lists.

    Inputs
    -------
    image_projection_head : torch.nn.Module
        The projection head to use for projecting the images.
    caption_projection_head : torch.nn.Module
        The projection to use for projecting the captions.
    embeddings_val_dataloader : torch.utils.data.DataLoader
        The dataloader to use for generating the base-embeddings data.

    Outputs
    -------
    image_proj_embeddings : torch.Tensor
        The Torch Tensor containing the image projections.
    caption_proj_embeddings : torch.Tensor
        The Torch Tensor containing the caption projections.
    captions_list : list
        The list of captions.
    sample_ids_list : list
        The list of sample ids.
    """

    image_model.to(device); caption_model.to(device)
    image_model.eval(); caption_model.eval()
    with torch.no_grad():
        image_proj_embeddings = torch.empty((len(dataloader.dataset), 256))
        caption_proj_embeddings = torch.empty((len(dataloader.dataset), 256))
        captions_list = []
        sample_ids_list = []
        for batch, (images_batch, captions_batch, captions, sample_ids) in enumerate(tqdm(dataloader, desc='Projecting Data')):
            images_batch = images_batch.to(device)
            if isinstance(captions_batch, dict):
                for item in captions_batch:
                  captions_batch[item] = captions_batch[item].to(device)
            else:
                captions_batch = captions_batch.to(device)
            image_proj_embeddings[batch*16:batch*16+len(images_batch), :] = image_model(images_batch)
            caption_proj_embeddings[batch*16:batch*16+len(images_batch), :] = caption_model(captions_batch)
            captions_list = captions_list + list(captions)
            sample_ids_list = sample_ids_list + list(sample_ids)
    return image_proj_embeddings, caption_proj_embeddings, captions_list, sample_ids_list


if __name__ == "__main__":
    download_ms_coco()
