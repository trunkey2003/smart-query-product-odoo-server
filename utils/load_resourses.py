import pickle
import faiss
import os


def load_image_embedding():
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_script_path, '..', 'dataset', 'feature', 'image_embedding.pkl')
    feature_path = os.path.normpath(path)
    with open(feature_path, 'rb') as fp:
        vector_embedding = pickle.load(fp)

    index = faiss.IndexFlatIP(vector_embedding.shape[1])
    index.add(vector_embedding)
    return index
