from dataclasses import dataclass


@dataclass
class SecurityConfig:
    EMBEDDING_DISTANCE_THRESHOLD = 0.5

    # Extraction of embeddings
    EMBEDDINGS_DF_PATH = "data/extracted_embeddings/allowed_embeddings.pkl"
    FACE_SAMPLES_FOLDER = "data/allowed_faces/"
    PREPROCESS_FACELOC_UPSAMPLING = 1
    PREPROCESS_FACELOC_MODEL = "hog"
    PREPROCESS_EXTRACTING_NUM_JITTERS = 10
