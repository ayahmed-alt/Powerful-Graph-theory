import os
import cv2
import numpy as np
import torch
import networkx as nx

from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Model Initialization
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet50(pretrained=True).to(DEVICE)
resnet.eval()
FEATURE_EXTRACTOR = torch.nn.Sequential(*list(resnet.children())[:-1])

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

YOLO_MODEL = YOLO("yolov8n.pt")


# =========================================================
# Feature Extraction
# =========================================================

def extract_features(image_crop: np.ndarray) -> np.ndarray:
    augmentations = [
        lambda x: x,
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_180),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
        lambda x: cv2.flip(x, 0),
        lambda x: cv2.flip(x, 1)
    ]

    features = []
    for aug in augmentations:
        img = aug(image_crop)
        tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = FEATURE_EXTRACTOR(tensor)
        features.append(feat.squeeze().cpu().numpy())

    return np.mean(features, axis=0)


# =========================================================
# Object Detection
# =========================================================

def detect_objects(image: np.ndarray, conf: float = 0.25) -> list:
    results = YOLO_MODEL(image, conf=conf)[0]
    return [tuple(map(int, box)) for box in results.boxes.xyxy.cpu().numpy()]


def detect_objects_with_patches(image: np.ndarray, conf: float = 0.25) -> list:
    h, w, _ = image.shape
    ph, pw = h // 3, w // 3
    boxes = []

    for i in range(3):
        for j in range(3):
            y1, y2 = i * ph, h if i == 2 else (i + 1) * ph
            x1, x2 = j * pw, w if j == 2 else (j + 1) * pw

            patch = image[y1:y2, x1:x2]
            results = YOLO_MODEL(patch, conf=conf)[0]

            for box in results.boxes.xyxy.cpu().numpy():
                bx1, by1, bx2, by2 = map(int, box)
                boxes.append((x1 + bx1, y1 + by1, x2 + bx2, y2 + by2))

    return boxes


# =========================================================
# Bounding Box Utilities
# =========================================================

def iou(box1, box2) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return 0.0 if union == 0 else inter / union


def merge_boxes(boxes, threshold=0.5) -> list:
    merged = []
    for box in boxes:
        if all(iou(box, b) < threshold for b in merged):
            merged.append(box)
    return merged


def hybrid_detect(image, conf=0.25, min_objects=2) -> list:
    boxes = detect_objects(image, conf)
    if len(boxes) < min_objects:
        boxes += detect_objects_with_patches(image, conf)
    return merge_boxes(boxes)


# =========================================================
# Graph Construction & Comparison
# =========================================================

def build_graph(image: np.ndarray, boxes: list) -> nx.Graph:
    G = nx.Graph()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        G.add_node(i, features=extract_features(crop), bbox=(x1, y1, x2, y2))

    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                ci = np.array(G.nodes[i]['bbox']).reshape(2, 2).mean(axis=1)
                cj = np.array(G.nodes[j]['bbox']).reshape(2, 2).mean(axis=1)
                G.add_edge(i, j, distance=np.linalg.norm(ci - cj))

    return G


def compare_graphs(G1: nx.Graph, G2: nx.Graph,
                   alpha: float = 0.6,
                   beta: float = 0.4) -> float:

    if not G1.nodes or not G2.nodes:
        return 0.0

    f1 = np.array([G1.nodes[n]['features'] for n in G1.nodes])
    f2 = np.array([G2.nodes[n]['features'] for n in G2.nodes])

    node_sim = cosine_similarity(f1, f2).max(axis=1).mean()

    D1 = nx.to_numpy_array(G1, weight="distance")
    D2 = nx.to_numpy_array(G2, weight="distance")

    size = min(D1.shape[0], D2.shape[0])
    edge_sim = 0.0

    if size > 1:
        edge_sim = cosine_similarity(
            D1[:size, :size].flatten().reshape(1, -1),
            D2[:size, :size].flatten().reshape(1, -1)
        )[0, 0]

    return alpha * node_sim + beta * edge_sim


# =========================================================
# Entry Point
# =========================================================

def main():
    pass  # Execution logic intentionally omitted for publication clarity


if __name__ == "__main__":
    main()

