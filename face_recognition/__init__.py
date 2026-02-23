"""
Face recognition using FaceNet (facenet-pytorch).

Detect faces with MTCNN, get 512-d embeddings with InceptionResnetV1 (VGGFace2),
compare faces by cosine similarity or L2 distance.
"""

from face_recognition.detector import (
    FaceRecognizer,
    detect_faces,
    get_embeddings,
    compare_faces,
    find_best_match,
    load_image,
)

__all__ = [
    "FaceRecognizer",
    "detect_faces",
    "get_embeddings",
    "compare_faces",
    "find_best_match",
    "load_image",
]
