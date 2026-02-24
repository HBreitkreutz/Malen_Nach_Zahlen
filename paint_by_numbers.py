#!/usr/bin/env python3
"""Create a paint-by-numbers template from an input image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import measure


@dataclass
class RegionInfo:
    color_index: int
    mask: np.ndarray
    area: int
    label_position: tuple[int, int]


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def quantize_colors(image: np.ndarray, n_colors: int, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = image.shape
    pixels = image.reshape((-1, 3)).astype(np.float32)

    model = KMeans(n_clusters=n_colors, n_init=10, random_state=random_state)
    labels = model.fit_predict(pixels)
    palette = np.clip(np.round(model.cluster_centers_), 0, 255).astype(np.uint8)
    return labels.reshape((h, w)), palette


def _choose_merge_target(
    component_mask: np.ndarray,
    labels: np.ndarray,
    current_label: int,
    palette: np.ndarray,
    component_color: np.ndarray,
) -> int:
    dilated = ndimage.binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool))
    border = dilated & ~component_mask
    neighbor_labels = labels[border]
    neighbor_labels = neighbor_labels[neighbor_labels != current_label]

    if neighbor_labels.size == 0:
        candidates = np.array([i for i in range(palette.shape[0]) if i != current_label], dtype=np.int32)
    else:
        candidates = np.unique(neighbor_labels)

    candidate_colors = palette[candidates].astype(np.float32)
    distances = np.linalg.norm(candidate_colors - component_color[None, :], axis=1)
    return int(candidates[np.argmin(distances)])


def enforce_min_region_size(
    labels: np.ndarray,
    palette: np.ndarray,
    min_region_pixels: int,
    rare_color_threshold_ratio: float,
    rare_color_preserve_components: int,
    max_iterations: int = 15,
) -> np.ndarray:
    labels = labels.copy()
    total_pixels = labels.size

    for _ in range(max_iterations):
        changed = False
        for color_index in range(palette.shape[0]):
            mask = labels == color_index
            if not mask.any():
                continue

            cc, count = ndimage.label(mask)
            component_areas = np.bincount(cc.ravel())[1:]
            preserved_components: set[int] = set()

            color_ratio = float(mask.sum()) / float(total_pixels)
            if rare_color_preserve_components > 0 and color_ratio <= rare_color_threshold_ratio:
                keep = min(rare_color_preserve_components, component_areas.size)
                if keep > 0:
                    largest_ids = np.argsort(component_areas)[-keep:] + 1
                    preserved_components = set(int(i) for i in largest_ids)

            for component_id in range(1, count + 1):
                component_mask = cc == component_id
                area = int(component_mask.sum())
                if component_id in preserved_components:
                    continue
                if area >= min_region_pixels:
                    continue

                component_color = palette[color_index].astype(np.float32)
                target = _choose_merge_target(component_mask, labels, color_index, palette, component_color)
                labels[component_mask] = target
                changed = True

        if not changed:
            break

    return labels


def collect_regions(labels: np.ndarray, palette: np.ndarray) -> list[RegionInfo]:
    regions: list[RegionInfo] = []
    for color_index in range(palette.shape[0]):
        color_mask = labels == color_index
        if not color_mask.any():
            continue

        cc, count = ndimage.label(color_mask)
        for component_id in range(1, count + 1):
            component_mask = cc == component_id
            area = int(component_mask.sum())
            y, x = best_label_position(component_mask)
            regions.append(
                RegionInfo(
                    color_index=color_index,
                    mask=component_mask,
                    area=area,
                    label_position=(x, y),
                )
            )

    return regions


def best_label_position(mask: np.ndarray) -> tuple[int, int]:
    # Distance-transform maximum gives an interior point farthest from boundaries.
    dist = ndimage.distance_transform_edt(mask)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(y), int(x)


def _mask_contours(mask: np.ndarray) -> Iterable[np.ndarray]:
    # Returns contours in (row, col) coordinates.
    return measure.find_contours(mask.astype(np.uint8), level=0.5)


def _draw_contour(pdf: canvas.Canvas, contour: np.ndarray, page_height: int, y_offset: int) -> None:
    if contour.shape[0] < 2:
        return

    path = pdf.beginPath()
    first_y, first_x = contour[0]
    path.moveTo(float(first_x), float(page_height - (first_y + y_offset)))

    for y, x in contour[1:]:
        path.lineTo(float(x), float(page_height - (y + y_offset)))

    path.close()
    pdf.drawPath(path, fill=0, stroke=1)


def _font_size_for_region(area: int, mask: np.ndarray) -> int:
    dist = ndimage.distance_transform_edt(mask)
    radius = float(np.max(dist))
    size_from_radius = int(max(6, min(24, radius * 1.8)))

    size_from_area = int(max(6, min(24, np.sqrt(area) * 0.45)))
    return max(6, min(size_from_radius, size_from_area, 24))


def write_pdf(
    output_path: Path,
    labels: np.ndarray,
    palette: np.ndarray,
    regions: list[RegionInfo],
    legend_columns: int = 6,
) -> None:
    height, width = labels.shape
    legend_rows = int(np.ceil(palette.shape[0] / legend_columns))
    legend_height = 28 + legend_rows * 30

    page_width = width
    page_height = height + legend_height
    y_offset = legend_height

    pdf = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(0.6)

    # White drawing area
    pdf.setFillColor(colors.white)
    pdf.rect(0, legend_height, width, height, stroke=0, fill=1)

    # Draw all region boundaries.
    for region in regions:
        for contour in _mask_contours(region.mask):
            _draw_contour(pdf, contour, page_height=page_height, y_offset=y_offset)

    # Draw region numbers.
    pdf.setFillColor(colors.black)
    for region in regions:
        x, y = region.label_position
        number = str(region.color_index + 1)
        font_size = _font_size_for_region(region.area, region.mask)
        pdf.setFont("Helvetica", font_size)
        text_width = pdf.stringWidth(number, "Helvetica", font_size)
        px = x - text_width / 2
        py = page_height - (y + y_offset) - font_size / 3
        pdf.drawString(px, py, number)

    # Legend area
    pdf.setFont("Helvetica", 10)
    pdf.drawString(10, legend_height - 14, "Farbleiste:")

    swatch_w = max(90, width // legend_columns)
    swatch_h = 22

    for idx, rgb in enumerate(palette):
        row = idx // legend_columns
        col = idx % legend_columns

        x = 10 + col * swatch_w
        y = legend_height - 24 - (row + 1) * swatch_h

        if x + 70 > width:
            continue

        r, g, b = [int(v) for v in rgb]
        pdf.setFillColor(colors.Color(r / 255, g / 255, b / 255))
        pdf.rect(x, y + 4, 16, 16, stroke=1, fill=1)

        pdf.setFillColor(colors.black)
        pdf.drawString(x + 22, y + 8, f"{idx + 1}: ({r}, {g}, {b})")

    pdf.save()


def build_template(
    input_path: Path,
    output_path: Path,
    n_colors: int,
    min_region_ratio: float,
    rare_color_threshold_ratio: float,
    rare_color_preserve_components: int,
) -> None:
    image = load_image(input_path)
    labels, palette = quantize_colors(image, n_colors=n_colors)

    min_region_pixels = int(max(1, round(image.shape[0] * image.shape[1] * min_region_ratio)))
    labels = enforce_min_region_size(
        labels,
        palette,
        min_region_pixels=min_region_pixels,
        rare_color_threshold_ratio=rare_color_threshold_ratio,
        rare_color_preserve_components=rare_color_preserve_components,
    )

    regions = collect_regions(labels, palette)
    write_pdf(output_path, labels, palette, regions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paint-by-numbers PDF template from an image.")
    parser.add_argument("input_image", type=Path, help="Path to source image (png, jpeg, ...)")
    parser.add_argument("output_pdf", type=Path, help="Path to generated PDF")
    parser.add_argument("--colors", type=int, default=12, help="Number of colors in the reduced palette")
    parser.add_argument(
        "--min-region-ratio",
        type=float,
        default=0.005,
        help="Minimum region area as ratio of full image area (e.g. 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--rare-color-threshold-ratio",
        type=float,
        default=0.02,
        help="Colors that cover less than this ratio are treated as rare (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--rare-color-preserve-components",
        type=int,
        default=2,
        help="For rare colors, preserve this many largest regions from merging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.colors < 2:
        raise ValueError("--colors must be >= 2")
    if not (0 < args.min_region_ratio < 1):
        raise ValueError("--min-region-ratio must be between 0 and 1")

    if not (0 <= args.rare_color_threshold_ratio < 1):
        raise ValueError("--rare-color-threshold-ratio must be between 0 (inclusive) and 1")
    if args.rare_color_preserve_components < 0:
        raise ValueError("--rare-color-preserve-components must be >= 0")

    build_template(
        args.input_image,
        args.output_pdf,
        args.colors,
        args.min_region_ratio,
        rare_color_threshold_ratio=args.rare_color_threshold_ratio,
        rare_color_preserve_components=args.rare_color_preserve_components,
    )


if __name__ == "__main__":
    main()
