import torch
import sys
import os
current_script_path = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_script_path, '..', 'model')
sys.path.append(os.path.normpath(model_file_path))
import encoder
import numpy as np
from PIL import Image

image_file_path = os.path.join(current_script_path, '..', '..', 'dataset', 'output')


image_path = os.listdir(os.path.normpath(image_file_path))
# just get the name of the image
# image_path = [os.path.normpath(image_file_path) + "\\" + path for path in image_id if '.jpg' in path]
image_id = [path for path in image_path if '.jpg' in path]
# remove the .jpg
image_id = [path.replace('.jpg', '') for path in image_id]

image_id.sort()

textEncoderHolder = encoder.Encoder()
#result is number of return result
def Search(search_text, results, index):
    with torch.no_grad():
        text_search_embedding = textEncoderHolder.encode_text([search_text], batch_size=128)
    text_search_embedding = text_search_embedding/np.linalg.norm(text_search_embedding, ord=2, axis=-1, keepdims=True)
    distances, indices = index.search(text_search_embedding.reshape(1, -1), results)
    distances = distances[0]
    indices = indices[0]

    indices_distances = list(zip(indices, distances))
    indices_distances.sort(key=lambda x: x[1])
    fixed_size = (300, 300)
    result_paths = []
    for idx, distance in indices_distances:
        result_paths.append(image_id[idx])
    return result_paths
    