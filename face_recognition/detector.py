"""
Face detection and recognition using FaceNet (facenet-pytorch).

Uses MTCNN for face detection and InceptionResnetV1 (pretrained VGGFace2) for 512-d embeddings.
"""

from pathlib import Path
from typing import Any

import torch


def load_image(path: str | Path) -> Any:
    """Load image as PIL Image. Required for MTCNN."""
    from PIL import Image
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


class FaceRecognizer:
    """
    Face detection (MTCNN) + FaceNet embeddings (InceptionResnetV1).
    """

    def __init__(self, device: str | None = None):
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError as e:
            raise ImportError(
                "Install facenet-pytorch: pip install facenet-pytorch"
            ) from e
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self._device,
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)

    def detect(self, image: Any) -> tuple[Any, list[float] | None]:
        """
        Detect faces in image (PIL or path). Returns (cropped_face_tensors, probs).
        cropped_face_tensors: (N, 3, 160, 160) on device, or None if no faces.
        """
        if isinstance(image, (str, Path)):
            image = load_image(image)
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return None, None
        face_tensors = self.mtcnn(image)
        if face_tensors is None:
            return None, None
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)
        return face_tensors.to(self._device), probs.tolist()

    def get_embeddings(self, image: Any) -> tuple[list[list[float]], list[float] | None]:
        """
        Detect faces and return 512-d FaceNet embeddings for each.
        Returns (list of 512-d vectors, list of detection probs or None).
        """
        face_tensors, probs = self.detect(image)
        if face_tensors is None:
            return [], None
        with torch.no_grad():
            embeddings = self.resnet(face_tensors)
        return embeddings.cpu().tolist(), probs

    def get_boxes(self, image: Any) -> tuple[list[tuple[float, ...]], list[float] | None]:
        """Detect faces and return bounding boxes (x1,y1,x2,y2) and probs."""
        if isinstance(image, (str, Path)):
            image = load_image(image)
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return [], None
        return [tuple(map(float, b)) for b in boxes], probs.tolist()


# --- Module-level helpers (lazy singleton) ---
_recognizer: FaceRecognizer | None = None


def _get_recognizer(device: str | None = None) -> FaceRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer(device=device)
    return _recognizer


def detect_faces(
    image_path: str | Path,
    *,
    device: str | None = None,
) -> tuple[list[tuple[float, ...]], list[float] | None]:
    """
    Detect faces in an image. Returns (list of boxes (x1,y1,x2,y2), list of probs).
    """
    r = _get_recognizer(device)
    return r.get_boxes(image_path)


def get_embeddings(
    image_path: str | Path,
    *,
    device: str | None = None,
) -> tuple[list[list[float]], list[float] | None]:
    """
    Get FaceNet 512-d embeddings for each face in the image.
    Returns (list of embeddings, list of detection probs or None).
    """
    r = _get_recognizer(device)
    return r.get_embeddings(image_path)


def compare_faces(
    embedding1: list[float],
    embedding2: list[float],
    metric: str = "cosine",
) -> float:
    """
    Compare two FaceNet embeddings. Returns similarity (cosine: 1 = same, 0 = orth; l2: 0 = same).
    """
    a = torch.tensor(embedding1, dtype=torch.float32)
    b = torch.tensor(embedding2, dtype=torch.float32)
    if metric == "cosine":
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        return float(a.dot(b).item())
    if metric == "l2" or metric == "euclidean":
        return float((a - b).norm().item())
    raise ValueError("metric must be 'cosine' or 'l2'")


def find_best_match(
    query_embedding: list[float],
    gallery_embeddings: list[list[float]],
    *,
    metric: str = "cosine",
) -> tuple[int, float]:
    """
    Find the gallery index with highest similarity to the query.
    Returns (best_index, score). For cosine, higher is better; for l2, lower is better.
    """
    if not gallery_embeddings:
        return -1, 0.0
    if metric == "cosine":
        best_i = 0
        best_sim = compare_faces(query_embedding, gallery_embeddings[0], metric="cosine")
        for i in range(1, len(gallery_embeddings)):
            sim = compare_faces(query_embedding, gallery_embeddings[i], metric="cosine")
            if sim > best_sim:
                best_sim = sim
                best_i = i
        return best_i, best_sim
    best_i = 0
    best_dist = compare_faces(query_embedding, gallery_embeddings[0], metric="l2")
    for i in range(1, len(gallery_embeddings)):
        d = compare_faces(query_embedding, gallery_embeddings[i], metric="l2")
        if d < best_dist:
            best_dist = d
            best_i = i
    return best_i, best_dist
