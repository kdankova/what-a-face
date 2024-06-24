import numpy as np
import pandas as pd
from face_recognition import face_distance
from config import SecurityConfig


class EmbeddingsManager:
    def __init__(self, embeddings_dataframe_path=SecurityConfig.EMBEDDINGS_DF_PATH):
        self.embeddings_df = pd.read_pickle(embeddings_dataframe_path)
        self.embeddings_df.embedding = self.embeddings_df.embedding.apply(np.array)

    def check_embedding(self, embedding):
        allowed_embeddings = self.embeddings_df.embedding
        distances = face_distance(list(allowed_embeddings.values), embedding)
        names = self.embeddings_df.name.values

        closest_id = np.argmin(distances)

        distance = distances[closest_id]
        match = bool(distance < SecurityConfig.EMBEDDING_DISTANCE_THRESHOLD)
        name = names[closest_id] if match else "Unknown"

        return match, name, distance
