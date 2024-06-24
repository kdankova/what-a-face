from pathlib import Path

import face_recognition as fr
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import SecurityConfig

# from tqdm.auto import tqdm  # for notebooks

tqdm.pandas()


def get_embedding_from_file(path):
    image = fr.load_image_file(path)
    face_locations = fr.face_locations(
        image,
        SecurityConfig.PREPROCESS_FACELOC_UPSAMPLING,
        SecurityConfig.PREPROCESS_FACELOC_MODEL,
    )
    embeddings = fr.face_encodings(image, face_locations, SecurityConfig.PREPROCESS_EXTRACTING_NUM_JITTERS)
    if len(embeddings) == 0:
        return np.nan
    return embeddings[0]


if __name__ == "__main__":
    extensions = ["jpg", "jpeg", "png"]
    files = []
    for ext in extensions:
        files += Path(SecurityConfig.FACE_SAMPLES_FOLDER).glob(f"*/*.{ext}")

    embeddings_df = pd.DataFrame({"files": files})
    embeddings_df["name"] = embeddings_df.files.apply(lambda f: f.parent.name)
    print("Extracting embeddings from images")
    embeddings_df["embedding"] = embeddings_df.files.progress_apply(get_embedding_from_file)
    embeddings_df = embeddings_df.dropna()
    embeddings_df = embeddings_df.groupby("name", as_index=False).mean(numeric_only=False)
    embeddings_df.to_pickle(SecurityConfig.EMBEDDINGS_DF_PATH)
    print(f"Embeddings saved to {SecurityConfig.EMBEDDINGS_DF_PATH}!")
