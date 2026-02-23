"""
CLI for face recognition (detect faces, get embeddings, compare two images).
Run: python -m face_recognition <image_path> [second_image_path]
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face recognition with FaceNet: detect faces, embeddings, compare two images."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to image (or first image when comparing).",
    )
    parser.add_argument(
        "second_image",
        type=Path,
        nargs="?",
        default=None,
        help="Optional second image to compare (cosine similarity of first face in each).",
    )
    parser.add_argument(
        "--boxes",
        action="store_true",
        help="Print bounding boxes and detection probabilities only.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for inference (default: auto).",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Error: not a file: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        from face_recognition import (
            detect_faces,
            get_embeddings,
            compare_faces,
        )
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install: pip install facenet-pytorch", file=sys.stderr)
        sys.exit(1)

    device = args.device

    if args.second_image is not None:
        if not args.second_image.is_file():
            print(f"Error: not a file: {args.second_image}", file=sys.stderr)
            sys.exit(1)
        emb1_list, _ = get_embeddings(args.image, device=device)
        emb2_list, _ = get_embeddings(args.second_image, device=device)
        if not emb1_list or not emb2_list:
            print("No face found in one or both images.", file=sys.stderr)
            sys.exit(1)
        sim = compare_faces(emb1_list[0], emb2_list[0], metric="cosine")
        print(f"Cosine similarity (first face): {sim:.4f}")
        return

    if args.boxes:
        boxes, probs = detect_faces(args.image, device=device)
        if not boxes:
            print("No faces detected.")
            return
        for i, (box, prob) in enumerate(zip(boxes, probs or [])):
            print(f"Face {i+1}: box={box} prob={prob:.3f}")
        return

    embeddings, probs = get_embeddings(args.image, device=device)
    if not embeddings:
        print("No faces detected.")
        return
    print(f"Detected {len(embeddings)} face(s). Embedding dim: {len(embeddings[0])}")
    for i, (emb, prob) in enumerate(zip(embeddings, probs or [])):
        print(f"  Face {i+1}: prob={prob:.3f}")


if __name__ == "__main__":
    main()
