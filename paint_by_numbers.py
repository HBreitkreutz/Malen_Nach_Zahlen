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
from skimage import color as skcolor
from skimage import filters
from skimage import measure
from skimage.segmentation import slic


@dataclass
class RegionInfo:
    color_index: int
    mask: np.ndarray
    area: int
    label_position: tuple[int, int]


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def resize_for_processing(image: np.ndarray, max_processing_dimension: int) -> np.ndarray:
    if max_processing_dimension <= 0:
        return image

    h, w, _ = image.shape
    current_max = max(h, w)
    if current_max <= max_processing_dimension:
        return image

    scale = float(max_processing_dimension) / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(image, mode="RGB")
    resized = pil.resize((new_w, new_h), resample=Image.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def quantize_colors(image: np.ndarray, n_colors: int, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = image.shape
    pixels = image.reshape((-1, 3)).astype(np.float32)

    model = KMeans(n_clusters=n_colors, n_init=10, random_state=random_state)
    labels = model.fit_predict(pixels)
    palette = np.clip(np.round(model.cluster_centers_), 0, 255).astype(np.uint8)
    return labels.reshape((h, w)), palette


def quantize_colors_shape_first(
    image: np.ndarray,
    n_colors: int,
    superpixels: int,
    superpixel_compactness: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = image.shape
    sp_labels = slic(
        image,
        n_segments=superpixels,
        compactness=superpixel_compactness,
        start_label=0,
        channel_axis=-1,
    )

    n_sp = int(sp_labels.max()) + 1
    sp_colors = np.zeros((n_sp, 3), dtype=np.float32)
    for sp_idx in range(n_sp):
        mask = sp_labels == sp_idx
        if not mask.any():
            continue
        sp_colors[sp_idx] = image[mask].mean(axis=0)

    model = KMeans(n_clusters=n_colors, n_init=10, random_state=random_state)
    sp_color_labels = model.fit_predict(sp_colors)
    palette = np.clip(np.round(model.cluster_centers_), 0, 255).astype(np.uint8)

    labels = np.zeros((h, w), dtype=np.int32)
    for sp_idx in range(n_sp):
        labels[sp_labels == sp_idx] = int(sp_color_labels[sp_idx])
    return labels, palette


def smooth_label_boundaries(labels: np.ndarray, n_labels: int, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return labels

    labels = labels.copy()
    kernel = np.ones((3, 3), dtype=np.int16)
    for _ in range(iterations):
        counts = []
        for label_idx in range(n_labels):
            mask = (labels == label_idx).astype(np.int16)
            counts.append(ndimage.convolve(mask, kernel, mode="nearest"))
        stacked = np.stack(counts, axis=0)
        labels = np.argmax(stacked, axis=0).astype(np.int32)
    return labels


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


def _component_neighbor_contrast(
    component_mask: np.ndarray,
    labels: np.ndarray,
    current_label: int,
    palette: np.ndarray,
) -> float:
    dilated = ndimage.binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool))
    border = dilated & ~component_mask
    neighbor_labels = labels[border]
    neighbor_labels = np.unique(neighbor_labels[neighbor_labels != current_label])
    if neighbor_labels.size == 0:
        return 0.0

    component_color = palette[current_label].astype(np.float32)
    neighbor_colors = palette[neighbor_labels].astype(np.float32)
    distances = np.linalg.norm(neighbor_colors - component_color[None, :], axis=1)
    return float(np.max(distances))


def _component_boundary_edge_strength(component_mask: np.ndarray, edge_strength_map: np.ndarray | None) -> float:
    if edge_strength_map is None:
        return 0.0
    dilated = ndimage.binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool))
    border = dilated & ~component_mask
    if not border.any():
        return 0.0
    return float(np.mean(edge_strength_map[border]))


def _component_is_thin_structure(component_mask: np.ndarray, area: int) -> bool:
    ys, xs = np.where(component_mask)
    if ys.size == 0:
        return False

    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    short_axis = max(1, min(h, w))
    long_axis = max(h, w)
    aspect_ratio = float(long_axis) / float(short_axis)

    radius = float(np.max(ndimage.distance_transform_edt(component_mask)))
    return area >= 8 and aspect_ratio >= 4.0 and radius <= 2.0


def enforce_min_region_size(
    labels: np.ndarray,
    palette: np.ndarray,
    min_region_pixels: int,
    rare_color_threshold_ratio: float,
    rare_color_preserve_components: int,
    preserve_contrast_threshold: float,
    edge_strength_map: np.ndarray | None,
    preserve_edge_strength_threshold: float,
    preserve_thin_structures: bool,
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
                if _component_neighbor_contrast(component_mask, labels, color_index, palette) >= preserve_contrast_threshold:
                    continue
                if _component_boundary_edge_strength(component_mask, edge_strength_map) >= preserve_edge_strength_threshold:
                    continue
                if preserve_thin_structures and _component_is_thin_structure(component_mask, area):
                    continue

                component_color = palette[color_index].astype(np.float32)
                target = _choose_merge_target(component_mask, labels, color_index, palette, component_color)
                labels[component_mask] = target
                changed = True

        if not changed:
            break

    return labels


def _component_aspect_ratio(component_mask: np.ndarray) -> float:
    ys, xs = np.where(component_mask)
    if ys.size == 0:
        return 1.0
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    short_axis = max(1, min(h, w))
    long_axis = max(h, w)
    return float(long_axis) / float(short_axis)


def _component_perimeter_area_ratio(component_mask: np.ndarray, area: int) -> float:
    if area <= 0:
        return 0.0
    eroded = ndimage.binary_erosion(component_mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = component_mask & ~eroded
    perimeter = int(boundary.sum())
    return float(perimeter) / float(area)


def _component_max_radius(component_mask: np.ndarray) -> float:
    return float(np.max(ndimage.distance_transform_edt(component_mask)))


def _is_unpaintable_component(
    component_mask: np.ndarray,
    area: int,
    min_paintable_radius: float,
    max_aspect_ratio: float,
    max_perimeter_area_ratio: float,
) -> bool:
    radius = _component_max_radius(component_mask)
    if radius >= min_paintable_radius:
        return False

    aspect_ratio = _component_aspect_ratio(component_mask)
    perimeter_area_ratio = _component_perimeter_area_ratio(component_mask, area)
    return aspect_ratio >= max_aspect_ratio or perimeter_area_ratio >= max_perimeter_area_ratio


def simplify_unpaintable_regions(
    labels: np.ndarray,
    palette: np.ndarray,
    min_paintable_radius: float,
    max_aspect_ratio: float,
    max_perimeter_area_ratio: float,
    max_merge_region_ratio: float,
    max_iterations: int = 8,
) -> np.ndarray:
    if min_paintable_radius <= 0:
        return labels

    labels = labels.copy()
    total_pixels = labels.size

    for _ in range(max_iterations):
        changed = False
        for color_index in range(palette.shape[0]):
            mask = labels == color_index
            if not mask.any():
                continue

            cc, count = ndimage.label(mask)
            for component_id in range(1, count + 1):
                component_mask = cc == component_id
                area = int(component_mask.sum())
                if area <= 0:
                    continue

                area_ratio = float(area) / float(total_pixels)
                if area_ratio > max_merge_region_ratio:
                    continue

                if not _is_unpaintable_component(
                    component_mask,
                    area=area,
                    min_paintable_radius=min_paintable_radius,
                    max_aspect_ratio=max_aspect_ratio,
                    max_perimeter_area_ratio=max_perimeter_area_ratio,
                ):
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


def _simplify_contour(contour: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0 or contour.shape[0] < 4:
        return contour
    simplified = measure.approximate_polygon(contour, tolerance=tolerance)
    if simplified.shape[0] < 2:
        return contour
    return simplified


def _draw_contour(pdf: canvas.Canvas, contour: np.ndarray, page_height: int, y_offset: int, close_path: bool = True) -> None:
    if contour.shape[0] < 2:
        return

    path = pdf.beginPath()
    first_y, first_x = contour[0]
    path.moveTo(float(first_x), float(page_height - (first_y + y_offset)))

    for y, x in contour[1:]:
        path.lineTo(float(x), float(page_height - (y + y_offset)))

    if close_path:
        path.close()
    pdf.drawPath(path, fill=0, stroke=1)


def _font_size_for_region(area: int, mask: np.ndarray) -> int:
    dist = ndimage.distance_transform_edt(mask)
    radius = float(np.max(dist))
    size_from_radius = int(max(6, min(24, radius * 1.8)))

    size_from_area = int(max(6, min(24, np.sqrt(area) * 0.45)))
    return max(6, min(size_from_radius, size_from_area, 24))


def _label_boundary_mask(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    return boundary


def collect_missing_detail_contours(
    image: np.ndarray,
    labels: np.ndarray,
    detail_edge_threshold: float,
    detail_min_contour_points: int,
    detail_max_contours: int,
) -> list[np.ndarray]:
    if detail_edge_threshold <= 0:
        return []

    edge_strength = filters.sobel(skcolor.rgb2gray(image))
    detail_edges = edge_strength >= detail_edge_threshold

    # Remove edges already represented by label transitions.
    existing_boundaries = _label_boundary_mask(labels)
    existing_boundaries = ndimage.binary_dilation(existing_boundaries, structure=np.ones((3, 3), dtype=bool))
    missing_edges = detail_edges & ~existing_boundaries
    missing_edges = ndimage.binary_closing(missing_edges, structure=np.ones((2, 2), dtype=bool))

    contours = measure.find_contours(missing_edges.astype(np.uint8), level=0.5)
    contours = [c for c in contours if c.shape[0] >= detail_min_contour_points]
    contours.sort(key=lambda c: int(c.shape[0]), reverse=True)
    if detail_max_contours > 0:
        contours = contours[:detail_max_contours]
    return contours


def write_pdf(
    output_path: Path,
    labels: np.ndarray,
    palette: np.ndarray,
    regions: list[RegionInfo],
    extra_detail_contours: list[np.ndarray],
    contour_simplify_tolerance: float,
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
            contour = _simplify_contour(contour, tolerance=contour_simplify_tolerance)
            _draw_contour(pdf, contour, page_height=page_height, y_offset=y_offset, close_path=True)

    # Draw edge-detail contours that are not represented by color-label boundaries.
    pdf.setLineWidth(0.35)
    for contour in extra_detail_contours:
        contour = _simplify_contour(contour, tolerance=contour_simplify_tolerance)
        _draw_contour(pdf, contour, page_height=page_height, y_offset=y_offset, close_path=False)
    pdf.setLineWidth(0.6)

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
    max_processing_dimension: int,
    boundary_smoothing_iterations: int,
    min_region_ratio: float,
    rare_color_threshold_ratio: float,
    rare_color_preserve_components: int,
    shape_first: bool,
    superpixels: int,
    superpixel_compactness: float,
    preserve_contrast_threshold: float,
    preserve_edge_strength_threshold: float,
    preserve_thin_structures: bool,
    simplify_paintability: bool,
    min_paintable_radius: float,
    max_aspect_ratio: float,
    max_perimeter_area_ratio: float,
    max_paintability_merge_ratio: float,
    detail_edge_threshold: float,
    detail_min_contour_points: int,
    detail_max_contours: int,
    contour_simplify_tolerance: float,
) -> None:
    image = load_image(input_path)
    image = resize_for_processing(image, max_processing_dimension=max_processing_dimension)
    edge_strength_map = filters.sobel(skcolor.rgb2gray(image))
    if shape_first:
        labels, palette = quantize_colors_shape_first(
            image,
            n_colors=n_colors,
            superpixels=superpixels,
            superpixel_compactness=superpixel_compactness,
        )
    else:
        labels, palette = quantize_colors(image, n_colors=n_colors)
    labels = smooth_label_boundaries(labels, n_labels=palette.shape[0], iterations=boundary_smoothing_iterations)

    min_region_pixels = int(max(1, round(image.shape[0] * image.shape[1] * min_region_ratio)))
    labels = enforce_min_region_size(
        labels,
        palette,
        min_region_pixels=min_region_pixels,
        rare_color_threshold_ratio=rare_color_threshold_ratio,
        rare_color_preserve_components=rare_color_preserve_components,
        preserve_contrast_threshold=preserve_contrast_threshold,
        edge_strength_map=edge_strength_map,
        preserve_edge_strength_threshold=preserve_edge_strength_threshold,
        preserve_thin_structures=preserve_thin_structures,
    )
    if simplify_paintability:
        labels = simplify_unpaintable_regions(
            labels,
            palette,
            min_paintable_radius=min_paintable_radius,
            max_aspect_ratio=max_aspect_ratio,
            max_perimeter_area_ratio=max_perimeter_area_ratio,
            max_merge_region_ratio=max_paintability_merge_ratio,
        )

    regions = collect_regions(labels, palette)
    extra_detail_contours = collect_missing_detail_contours(
        image=image,
        labels=labels,
        detail_edge_threshold=detail_edge_threshold,
        detail_min_contour_points=detail_min_contour_points,
        detail_max_contours=detail_max_contours,
    )
    write_pdf(
        output_path,
        labels,
        palette,
        regions,
        extra_detail_contours=extra_detail_contours,
        contour_simplify_tolerance=contour_simplify_tolerance,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paint-by-numbers PDF template from an image.")
    parser.add_argument("input_image", type=Path, help="Path to source image (png, jpeg, ...)")
    parser.add_argument("output_pdf", type=Path, help="Path to generated PDF")
    parser.add_argument("--colors", type=int, default=12, help="Number of colors in the reduced palette")
    parser.add_argument(
        "--max-processing-dimension",
        type=int,
        default=1400,
        help="Downscale input so max(width, height) is this size before segmentation (0 disables)",
    )
    parser.add_argument(
        "--boundary-smoothing-iterations",
        type=int,
        default=0,
        help="How many 3x3 majority-smoothing passes to apply on label boundaries (0 disables)",
    )
    parser.add_argument(
        "--shape-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Segment image into superpixels before color quantization (default: enabled)",
    )
    parser.add_argument(
        "--superpixels",
        type=int,
        default=1800,
        help="Target number of superpixels when --shape-first is enabled",
    )
    parser.add_argument(
        "--superpixel-compactness",
        type=float,
        default=10.0,
        help="SLIC compactness (higher = blockier superpixels)",
    )
    parser.add_argument(
        "--min-region-ratio",
        type=float,
        default=0.002,
        help="Minimum region area as ratio of full image area (e.g. 0.002 = 0.2%%)",
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
    parser.add_argument(
        "--preserve-contrast-threshold",
        type=float,
        default=35.0,
        help="Protect tiny regions from merging if neighbor color contrast exceeds this RGB distance",
    )
    parser.add_argument(
        "--preserve-edge-strength-threshold",
        type=float,
        default=0.04,
        help="Protect tiny regions when boundary edge strength in source image exceeds this value",
    )
    parser.add_argument(
        "--preserve-thin-structures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Protect thin elongated regions (e.g. strings, whiskers) from merging (default: enabled)",
    )
    parser.add_argument(
        "--simplify-paintability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify regions that are too thin/frayed to be realistically paintable (default: enabled)",
    )
    parser.add_argument(
        "--min-paintable-radius",
        type=float,
        default=1.8,
        help="Minimum interior radius (in pixels) for paintable regions; lower values keep thinner regions",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=6.0,
        help="Unpaintable simplification trigger for elongated regions when thin",
    )
    parser.add_argument(
        "--max-perimeter-area-ratio",
        type=float,
        default=1.3,
        help="Unpaintable simplification trigger for very frayed/thin boundaries when thin",
    )
    parser.add_argument(
        "--max-paintability-merge-ratio",
        type=float,
        default=0.04,
        help="Do not simplify very large regions above this area ratio",
    )
    parser.add_argument(
        "--detail-edge-threshold",
        type=float,
        default=0.085,
        help="Draw additional contours from source edges not covered by color boundaries (0 disables)",
    )
    parser.add_argument(
        "--detail-min-contour-points",
        type=int,
        default=28,
        help="Minimum contour length (in contour points) for additional detail edges",
    )
    parser.add_argument(
        "--detail-max-contours",
        type=int,
        default=0,
        help="Maximum number of additional detail contours to draw (0 = unlimited)",
    )
    parser.add_argument(
        "--contour-simplify-tolerance",
        type=float,
        default=1.4,
        help="Douglas-Peucker tolerance in pixels for smoothing drawn contours (0 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.colors < 2:
        raise ValueError("--colors must be >= 2")
    if args.max_processing_dimension < 0:
        raise ValueError("--max-processing-dimension must be >= 0")
    if args.boundary_smoothing_iterations < 0:
        raise ValueError("--boundary-smoothing-iterations must be >= 0")
    if not (0 < args.min_region_ratio < 1):
        raise ValueError("--min-region-ratio must be between 0 and 1")

    if not (0 <= args.rare_color_threshold_ratio < 1):
        raise ValueError("--rare-color-threshold-ratio must be between 0 (inclusive) and 1")
    if args.rare_color_preserve_components < 0:
        raise ValueError("--rare-color-preserve-components must be >= 0")
    if args.superpixels < 50:
        raise ValueError("--superpixels must be >= 50")
    if args.superpixel_compactness <= 0:
        raise ValueError("--superpixel-compactness must be > 0")
    if args.preserve_contrast_threshold < 0:
        raise ValueError("--preserve-contrast-threshold must be >= 0")
    if args.preserve_edge_strength_threshold < 0:
        raise ValueError("--preserve-edge-strength-threshold must be >= 0")
    if args.min_paintable_radius < 0:
        raise ValueError("--min-paintable-radius must be >= 0")
    if args.max_aspect_ratio < 1:
        raise ValueError("--max-aspect-ratio must be >= 1")
    if args.max_perimeter_area_ratio < 0:
        raise ValueError("--max-perimeter-area-ratio must be >= 0")
    if not (0 <= args.max_paintability_merge_ratio <= 1):
        raise ValueError("--max-paintability-merge-ratio must be between 0 and 1")
    if args.detail_edge_threshold < 0:
        raise ValueError("--detail-edge-threshold must be >= 0")
    if args.detail_min_contour_points < 2:
        raise ValueError("--detail-min-contour-points must be >= 2")
    if args.detail_max_contours < 0:
        raise ValueError("--detail-max-contours must be >= 0")
    if args.contour_simplify_tolerance < 0:
        raise ValueError("--contour-simplify-tolerance must be >= 0")

    build_template(
        args.input_image,
        args.output_pdf,
        args.colors,
        args.max_processing_dimension,
        args.boundary_smoothing_iterations,
        args.min_region_ratio,
        rare_color_threshold_ratio=args.rare_color_threshold_ratio,
        rare_color_preserve_components=args.rare_color_preserve_components,
        shape_first=args.shape_first,
        superpixels=args.superpixels,
        superpixel_compactness=args.superpixel_compactness,
        preserve_contrast_threshold=args.preserve_contrast_threshold,
        preserve_edge_strength_threshold=args.preserve_edge_strength_threshold,
        preserve_thin_structures=args.preserve_thin_structures,
        simplify_paintability=args.simplify_paintability,
        min_paintable_radius=args.min_paintable_radius,
        max_aspect_ratio=args.max_aspect_ratio,
        max_perimeter_area_ratio=args.max_perimeter_area_ratio,
        max_paintability_merge_ratio=args.max_paintability_merge_ratio,
        detail_edge_threshold=args.detail_edge_threshold,
        detail_min_contour_points=args.detail_min_contour_points,
        detail_max_contours=args.detail_max_contours,
        contour_simplify_tolerance=args.contour_simplify_tolerance,
    )


if __name__ == "__main__":
    main()
