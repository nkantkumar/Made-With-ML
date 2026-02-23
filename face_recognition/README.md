# Face Recognition (FaceNet)

Face detection and recognition using **FaceNet** via [facenet-pytorch](https://github.com/timesler/facenet-pytorch): MTCNN for detection and InceptionResnetV1 (VGGFace2) for 512-d embeddings.

## Setup

**Python 3.11 or 3.12 required.** On Python 3.14, `facenet-pytorch`’s Pillow dependency does not build (no wheel; source build fails). Use a venv with 3.11/3.12 for this package.

From the repo root:

```bash
# With Python 3.11 or 3.12 active (e.g. pyenv shell 3.12, or a venv):
pip install -r face_recognition/requirements-face.txt
```

Or from the `face_recognition` folder:

```bash
pip install -r requirements-face.txt
```

This installs `facenet-pytorch` (and PyTorch). For GPU support, install a CUDA build of PyTorch if needed.

## Usage

### CLI

- **Detect faces and show embeddings** (default):
  ```bash
  python -m face_recognition path/to/image.jpg
  ```
- **Show bounding boxes and detection probabilities only**:
  ```bash
  python -m face_recognition path/to/image.jpg --boxes
  ```
- **Compare two images** (cosine similarity of the first face in each):
  ```bash
  python -m face_recognition path/to/face1.jpg path/to/face2.jpg
  ```

### Python API

```python
from face_recognition import (
    FaceRecognizer,
    detect_faces,
    get_embeddings,
    compare_faces,
    find_best_match,
    load_image,
)

# Option 1: module-level helpers (lazy singleton)
boxes, probs = detect_faces("photo.jpg")
embeddings, probs = get_embeddings("photo.jpg")

# Compare two faces (cosine: 1 = same, 0 = orthogonal)
sim = compare_faces(emb1, emb2, metric="cosine")

# Find best match in a gallery
best_idx, score = find_best_match(query_emb, gallery_embeddings, metric="cosine")

# Option 2: explicit recognizer (e.g. for device control)
recognizer = FaceRecognizer(device="cpu")
embeddings, probs = recognizer.get_embeddings("photo.jpg")
```

## Notes

- First run downloads MTCNN and InceptionResnetV1 (VGGFace2) weights.
- Images can be paths or PIL `Image` (RGB).
- Embeddings are 512-d; use cosine similarity or L2 distance for comparison.
