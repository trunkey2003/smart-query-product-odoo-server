import numpy as np
import os
import pickle
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_script_path, '..', 'model')
sys.path.append(os.path.normpath(model_file_path))

import encoder

encoderHolder = encoder.Encoder()

image_file_path = os.path.join(current_script_path, '..', '..', 'dataset', 'output')


image_path = os.listdir(os.path.normpath(image_file_path))
image_path = [os.path.normpath(image_file_path) + "\\" + path for path in image_path if '.jpg' in path]
image_path.sort()

print(image_path)

vector_embedding = np.array(encoderHolder.encode_images(image_path,128))

path = os.path.join(current_script_path, '..', 'dataset', 'feature')
feature_file_path = os.path.normpath(path)

output_path = os.path.join(feature_file_path, 'image_embedding.pkl')

if not os.path.exists(output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(vector_embedding, f)
    print(f"Image embeddings saved to {output_path}")
else:
    print(f"File {output_path} already exists. Skipping the save operation.")