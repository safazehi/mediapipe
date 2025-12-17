"""Gesture pipeline prototype following reference architecture.

This module wires a simple, traceable implementation of the stages shown in the
architecture diagram:

1. Frame buffering + time indexing
2. Shared preprocessing (MediaPipe + RGB keyframe extraction)
3. Sliding window generation
4. Isolated gesture classification
5. Per-window class probabilities
6. Temporal smoothing + boundary decoding
7. Standardisation (train only)
8. Output gesture segments

The code intentionally remains lightweight: it focuses on clean data flow and
extensibility rather than specific model weights. Stubbed hooks are provided to
connect MediaPipe detectors or custom torch/tflite models.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np


@dataclass
class Frame:
    """Container for a single RGB frame with timestamp metadata."""

    image: np.ndarray
    timestamp: float  # Seconds since stream start


class Preprocessor(Protocol):
    """Protocol for preprocessing frames into model-ready keypoints/features."""

    def __call__(self, frames: Sequence[Frame]) -> np.ndarray:
        ...


class Classifier(Protocol):
    """Protocol for producing gesture logits or probabilities."""

    def __call__(self, features: np.ndarray) -> Dict[str, float]:
        ...


@dataclass
class FixedLagFrameBuffer:
    """Maintains a fixed-length frame buffer with time indexing."""

    max_seconds: float
    buffer: Deque[Frame] = field(default_factory=deque)

    def add(self, frame: Frame) -> None:
        self.buffer.append(frame)
        self._evict_old(frame.timestamp)

    def window(self, end_time: float, duration: float) -> List[Frame]:
        start_time = end_time - duration
        return [f for f in self.buffer if f.timestamp >= start_time]

    def _evict_old(self, latest_timestamp: float) -> None:
        threshold = latest_timestamp - self.max_seconds
        while self.buffer and self.buffer[0].timestamp < threshold:
            self.buffer.popleft()


@dataclass
class SlidingWindowGenerator:
    """Yields overlapping windows from the frame buffer."""

    window_seconds: float
    hop_seconds: float

    def windows(self, frames: Iterable[Frame]) -> Iterable[Tuple[float, float, List[Frame]]]:
        frames_sorted = sorted(frames, key=lambda f: f.timestamp)
        if not frames_sorted:
            return

        stream_start = frames_sorted[0].timestamp
        stream_end = frames_sorted[-1].timestamp
        current = stream_start

        while current + self.window_seconds <= stream_end + 1e-6:
            window_end = current + self.window_seconds
            window_frames = [f for f in frames_sorted if current <= f.timestamp < window_end]
            if window_frames:
                yield current, window_end, window_frames
            current += self.hop_seconds


@dataclass
class TemporalDecoder:
    """Applies smoothing and decodes gesture boundaries across windows."""

    smoothing: float = 0.6
    min_consecutive: int = 3
    background_label: str = "background"

    def decode(
        self, window_scores: List[Tuple[float, float, Dict[str, float]]]
    ) -> List[Tuple[str, float, float]]:
        """Return segments as (label, start_time, end_time)."""

        if not window_scores:
            return []

        window_ranges = [(start, end) for start, end, _ in window_scores]
        smoothed = self._smooth([scores for _, _, scores in window_scores])
        segments: List[Tuple[str, float, float]] = []
        active_label: Optional[str] = None
        start_time: Optional[float] = None
        last_end_time: Optional[float] = None

        for (win_start, win_end), scores in zip(window_ranges, smoothed):
            label, score = max(scores.items(), key=lambda item: item[1])
            last_end_time = win_end
            if label == self.background_label or score <= 0.5:
                if active_label is not None and start_time is not None:
                    segments.append((active_label, float(start_time), float(win_start)))
                    active_label, start_time = None, None
                continue

            if active_label is None:
                active_label, start_time = label, win_start
            elif label != active_label:
                segments.append((active_label, float(start_time), float(win_start)))
                active_label, start_time = label, win_start

        if active_label is not None and start_time is not None and last_end_time is not None:
            segments.append((active_label, float(start_time), float(last_end_time)))

        return segments

    def _smooth(self, window_scores: List[Dict[str, float]]) -> List[Dict[str, float]]:
        smoothed: List[Dict[str, float]] = []
        history: Deque[Dict[str, float]] = deque(maxlen=self.min_consecutive)
        for scores in window_scores:
            history.append(scores)
            averaged: Dict[str, float] = {}
            keys = set().union(*history)
            for key in keys:
                averaged[key] = sum(item.get(key, 0.0) for item in history) / len(history)
            smoothed.append(averaged)
        return smoothed


@dataclass
class GesturePipeline:
    """Full gesture recognition pipeline aligning with the provided architecture."""

    buffer: FixedLagFrameBuffer
    preprocessor: Preprocessor
    classifier: Classifier
    window_generator: SlidingWindowGenerator
    decoder: TemporalDecoder

    def process(self, new_frames: Sequence[Frame]) -> List[Tuple[str, float, float]]:
        for frame in new_frames:
            self.buffer.add(frame)

        recent_frames = list(self.buffer.buffer)
        window_scores: List[Tuple[float, float, Dict[str, float]]] = []

        for window_start, window_end, window_frames in self.window_generator.windows(
            recent_frames
        ):
            features = self.preprocessor(window_frames)
            scores = self.classifier(features)
            window_scores.append((window_start, window_end, scores))

        return self.decoder.decode(window_scores)


# --- Reference implementations for quick testing ---


def rgb_center_crop_preprocessor(frames: Sequence[Frame], crop_ratio: float = 0.8) -> np.ndarray:
    """Example preprocessor: naive center crop + temporal stacking."""

    processed = []
    for frame in frames:
        h, w, _ = frame.image.shape
        dh, dw = int(h * crop_ratio / 2), int(w * crop_ratio / 2)
        center_h, center_w = h // 2, w // 2
        cropped = frame.image[center_h - dh : center_h + dh, center_w - dw : center_w + dw]
        resized = np.resize(cropped, (112, 112, 3))
        processed.append(resized)
    return np.stack(processed, axis=0)


def toy_classifier(features: np.ndarray) -> Dict[str, float]:
    """A deterministic classifier that prefers motion-heavy windows."""

    # Use simple magnitude-based heuristic as placeholder for real model outputs.
    energy = float(np.mean(np.abs(np.diff(features, axis=0)))) if len(features) > 1 else 0.0
    confidence = min(1.0, energy / 50.0)
    return {"wave": confidence, "background": 1.0 - confidence}


def build_default_pipeline(max_buffer_seconds: float = 5.0) -> GesturePipeline:
    """Construct a ready-to-run pipeline using reference components."""

    buffer = FixedLagFrameBuffer(max_seconds=max_buffer_seconds)
    window_generator = SlidingWindowGenerator(window_seconds=1.0, hop_seconds=0.5)
    decoder = TemporalDecoder()
    return GesturePipeline(
        buffer=buffer,
        preprocessor=lambda frames: rgb_center_crop_preprocessor(frames),
        classifier=toy_classifier,
        window_generator=window_generator,
        decoder=decoder,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    pipeline = build_default_pipeline()
    # Create a fake stream with alternating motion and static content.
    frames = []
    for i in range(20):
        img = np.ones((128, 128, 3), dtype=np.float32) * i
        frames.append(Frame(image=img, timestamp=i * 0.2))

    segments = pipeline.process(frames)
    for label, start, end in segments:
        print(f"Detected {label} from {start:.1f}s to {end:.1f}s")
