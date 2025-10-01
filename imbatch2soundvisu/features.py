"""Feature extraction helpers for image sequencing experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

from PIL import Image, ImageFilter, ImageStat

from .datasets import ImageRecord

FeatureFunction = Callable[[Image.Image], float]


@dataclass(slots=True)
class FeatureExtractor:
    """Compute a set of scalar features for a collection of images."""

    features: dict[str, FeatureFunction]

    def process(self, records: Iterable[ImageRecord]) -> list[ImageRecord]:
        processed: list[ImageRecord] = []
        for record in records:
            with Image.open(record.path) as image:
                image = image.convert("RGB")
                record.features = {
                    name: float(function(image))
                    for name, function in self.features.items()
                }
            processed.append(record)
        return processed


def average_luminance(image: Image.Image) -> float:
    """Return the perceived brightness of ``image`` in the range ``[0, 255]``."""

    stat = ImageStat.Stat(image.convert("L"))
    return float(stat.mean[0])


def colorfulness(image: Image.Image) -> float:
    """Approximate how colorful an image is using channel variance."""

    stat = ImageStat.Stat(image)
    variance_sum = sum(stat.var)
    return float(math.sqrt(max(variance_sum, 0.0)))


def dominant_hue(image: Image.Image) -> float:
    """Return the average hue in degrees (``0``–``360``)."""

    hsv = image.convert("HSV")
    stat = ImageStat.Stat(hsv)
    hue = stat.mean[0]
    return float(hue * 360.0 / 255.0)


def saturation(image: Image.Image) -> float:
    """Average saturation in the range ``[0, 1]``."""

    hsv = image.convert("HSV")
    stat = ImageStat.Stat(hsv)
    return float(stat.mean[1] / 255.0)


def warmth(image: Image.Image) -> float:
    """Positive values lean red, negative values lean blue."""

    stat = ImageStat.Stat(image)
    r, g, b = stat.mean
    return float(((r + g) / 2) - b)


def aspect_ratio(image: Image.Image) -> float:
    """Width divided by height."""

    width, height = image.size
    if height == 0:
        return 0.0
    return float(width / height)


def edge_density(image: Image.Image) -> float:
    """Fraction of pixels containing detected edges."""

    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    threshold = 32
    edge_pixels = sum(histogram[threshold:])
    return float(edge_pixels / total)


def subject_scale(image: Image.Image) -> float:
    """Approximate portion of the frame occupied by the subject.

    The heuristic relies on the bounding box of edge responses.  Images with a
    tight crop around the subject tend to produce values closer to ``1``.
    """

    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    mask = edges.point(lambda value: 255 if value > 48 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return 0.0
    x0, y0, x1, y1 = bbox
    width, height = image.size
    area = (x1 - x0) * (y1 - y0)
    total_area = max(width * height, 1)
    return float(area / total_area)


DEFAULT_FEATURES: dict[str, FeatureFunction] = {
    "luminance": average_luminance,
    "colorfulness": colorfulness,
    "hue": dominant_hue,
    "saturation": saturation,
    "warmth": warmth,
    "aspect_ratio": aspect_ratio,
    "edge_density": edge_density,
    "subject_scale": subject_scale,
}
