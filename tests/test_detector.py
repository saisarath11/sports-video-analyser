"""
Unit tests for Object Detection and BoundingBox logic.
"""

import numpy as np
from core.models import BoundingBox, Detection
from core.detector import PlayerDetector


def test_bounding_box():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=100.0)
    assert bbox.width == 40.0
    assert bbox.height == 80.0
    assert bbox.center == (30.0, 60.0)
    assert bbox.area == 3200.0


def test_player_detector_initialization():
    detector = PlayerDetector()
    assert detector is not None


def test_detector_blank_frame():
    detector = PlayerDetector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_frame)
    assert isinstance(detections, list)
