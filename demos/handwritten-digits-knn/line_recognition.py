"""Tách và nhận diện ký tự trên một dòng bằng xử lý ảnh truyền thống."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from knn import ManualKNN
from preprocessing import load_grayscale_image, preprocess_character_image_array


@dataclass(frozen=True)
class LineRecognitionResult:
    """Kết quả một dòng cùng các khung ký tự theo thứ tự đọc."""

    text: str
    boxes: tuple[tuple[int, int, int, int], ...]
    region_predictions: tuple[str, ...] = ()


def _remove_horizontal_rules(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.copy()
    minimum_run = max(24, int(mask.shape[1] * 0.6))
    row_runs: dict[int, list[tuple[int, int]]] = {}
    for row_index, row in enumerate(mask):
        if int(row.sum()) < max(18, int(mask.shape[1] * 0.25)):
            continue
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        long_runs = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if end - start >= minimum_run
        ]
        if long_runs:
            row_runs[row_index] = long_runs

    candidate_rows = sorted(row_runs)
    groups: list[list[int]] = []
    for row_index in candidate_rows:
        if not groups or row_index > groups[-1][-1] + 1:
            groups.append([row_index])
        else:
            groups[-1].append(row_index)

    for group in groups:
        if len(group) > 2:
            continue
        for row_index in group:
            for start, end in row_runs[row_index]:
                cleaned[row_index, start:end] = False
    return cleaned


def _remove_repeated_edge_artifacts(mask: np.ndarray) -> np.ndarray:
    """Bỏ các vạch mảnh lặp sát mép; giữ nguyên nét đơn lẻ chưa rõ là nhiễu."""
    height, width = mask.shape
    edge_band = max(4, int(width * 0.02))
    maximum_stroke_width = max(2, int(np.ceil(width * 0.01)))
    minimum_stroke_height = max(4, int(height * 0.03))
    maximum_stroke_height = max(minimum_stroke_height, int(height * 0.15))
    maximum_x_difference = max(2, int(width * 0.005))
    candidates: dict[str, list[tuple[int, int, int, int, int]]] = {
        "left": [],
        "right": [],
    }

    for component in _connected_components(mask):
        x0, y0, x1, y1, _ = component
        component_width = x1 - x0
        component_height = y1 - y0
        if not (
            component_width <= maximum_stroke_width
            and minimum_stroke_height <= component_height <= maximum_stroke_height
        ):
            continue
        if x0 <= edge_band:
            candidates["left"].append(component)
        elif width - x1 <= edge_band:
            candidates["right"].append(component)

    removed: list[tuple[int, int, int, int, int]] = []
    for side, components in candidates.items():
        components.sort(key=lambda item: (item[0], item[1]))
        groups: list[list[tuple[int, int, int, int, int]]] = []
        for component in components:
            x_center = (component[0] + component[2]) / 2
            if not groups or abs(
                x_center
                - (groups[-1][-1][0] + groups[-1][-1][2]) / 2
            ) > maximum_x_difference:
                groups.append([component])
            else:
                groups[-1].append(component)

        for group in groups:
            y_positions = [component[1] for component in group]
            y_ends = [component[3] for component in group]
            repeated_over_line = (
                len(group) >= 4
                and max(y_ends) - min(y_positions) >= height * 0.25
            )
            if repeated_over_line:
                removed.extend(group)

    cleaned = mask.copy()
    for x0, y0, x1, y1, _ in removed:
        cleaned[y0:y1, x0:x1] = False
    return cleaned


def _otsu_threshold(image: np.ndarray) -> int:
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = image.size
    weighted_total = float(np.dot(np.arange(256), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    threshold = 0
    for value in range(256):
        background_weight += histogram[value]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += value * histogram[value]
        mean_background = background_sum / background_weight
        mean_foreground = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (
            mean_background - mean_foreground
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = value
    return threshold


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    contrast = 255 - image if float(np.median(border)) > 127 else image.copy()
    background_radius = max(3.0, min(image.shape) * 0.04)
    background = np.asarray(
        Image.fromarray(contrast).filter(
            ImageFilter.GaussianBlur(radius=background_radius)
        ),
        dtype=np.int16,
    )
    foreground_intensity = contrast.astype(np.int16) - background
    if int(foreground_intensity.max()) <= 8:
        return np.zeros_like(image, dtype=bool)

    corrected = np.clip(
        foreground_intensity + int(np.median(background)), 0, 255
    ).astype(np.uint8)
    threshold = max(8, _otsu_threshold(corrected))
    mask = _remove_horizontal_rules(corrected > threshold)
    return _remove_repeated_edge_artifacts(mask)


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    min_area = max(2, int(mask.size * 0.00002))

    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        if area >= min_area:
            components.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return components


def _merge_detached_dots(
    components: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(components) < 2:
        return [component[:4] for component in components]

    heights = np.asarray([y1 - y0 for _, y0, _, y1, _ in components])
    areas = np.asarray([area for _, _, _, _, area in components])
    initial_height = float(np.median(heights))
    initial_area = float(np.median(areas))
    body_indices = [
        index
        for index, (height, area) in enumerate(zip(heights, areas))
        if height >= initial_height * 0.5 and area >= initial_area * 0.2
    ]
    if not body_indices:
        body_indices = list(range(len(components)))
    median_height = float(np.median(heights[body_indices]))
    median_area = float(np.median(areas[body_indices]))
    merged_components = [list(component[:4]) for component in components]
    consumed: set[int] = set()

    for dot_index, (x0, y0, x1, y1, area) in enumerate(components):
        dot_width = x1 - x0
        dot_height = y1 - y0
        if dot_height > median_height * 0.45 or area > median_area * 0.35:
            continue
        candidates: list[tuple[float, int]] = []
        dot_center = (x0 + x1) / 2
        for body_index, (body_x0, body_y0, body_x1, body_y1, body_area) in enumerate(
            components
        ):
            if body_index == dot_index or body_index in consumed or body_y0 < y1:
                continue
            body_height = body_y1 - body_y0
            if body_height < median_height * 0.55 or body_area <= area:
                continue
            gap = body_y0 - y1
            body_center = (body_x0 + body_x1) / 2
            horizontal_distance = abs(dot_center - body_center)
            if gap > median_height * 0.8:
                continue
            if horizontal_distance > max(dot_width, body_x1 - body_x0) * 0.75 + 1:
                continue
            candidates.append((gap + horizontal_distance, body_index))
        if candidates:
            _, body_index = min(candidates)
            body = merged_components[body_index]
            body[0] = min(body[0], x0)
            body[1] = min(body[1], y0)
            body[2] = max(body[2], x1)
            body[3] = max(body[3], y1)
            consumed.add(dot_index)

    kept: list[tuple[int, int, int, int]] = []
    for index, component in enumerate(merged_components):
        if index in consumed:
            continue
        x0, y0, x1, y1 = component
        height = y1 - y0
        width = x1 - x0
        area = components[index][4]
        is_speck = (
            area <= max(4, median_area * 0.025)
            and height <= median_height * 0.18
            and width <= median_height * 0.18
        )
        could_be_detached_dot = any(
            body_index != index
            and components[body_index][1] >= y1
            and components[body_index][3] - components[body_index][1]
            >= median_height * 0.55
            and components[body_index][1] - y1 <= median_height * 1.2
            and abs(
                (x0 + x1) / 2
                - (components[body_index][0] + components[body_index][2]) / 2
            )
            <= max(width, components[body_index][2] - components[body_index][0])
            for body_index in body_indices
        )
        if not is_speck or could_be_detached_dot:
            kept.append(tuple(component))
    return kept


def _split_at_clear_valleys(
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return boxes
    median_height = float(np.median([y1 - y0 for _, y0, _, y1 in boxes]))
    median_width = (
        float(np.median([x1 - x0 for x0, _, x1, _ in boxes]))
        if len(boxes) > 1
        else median_height * 0.75
    )
    split_boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        width = x1 - x0
        height = y1 - y0
        if width < median_width * 1.5 or width < height * 1.05:
            split_boxes.append((x0, y0, x1, y1))
            continue
        projection = mask[y0:y1, x0:x1].sum(axis=0)
        lower = max(1, int(height * 0.12))
        upper = max(lower + 1, int(height * 0.35))
        candidates = np.arange(lower, max(lower + 1, width - upper))
        if candidates.size == 0:
            split_boxes.append((x0, y0, x1, y1))
            continue
        valley = int(candidates[np.argmin(projection[candidates])])
        valley_limit = max(1, int(height * 0.1))
        if projection[valley] > valley_limit:
            split_boxes.append((x0, y0, x1, y1))
            continue
        if projection[:valley].sum() < mask[y0:y1, x0:x1].sum() * 0.15:
            split_boxes.append((x0, y0, x1, y1))
            continue
        if projection[valley + 1 :].sum() < mask[y0:y1, x0:x1].sum() * 0.15:
            split_boxes.append((x0, y0, x1, y1))
            continue
        split_boxes.extend(
            ((x0, y0, x0 + valley, y1), (x0 + valley, y0, x1, y1))
        )
    return split_boxes


def _analyze_line(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Ảnh dòng phải là mảng grayscale 2D không rỗng.")
    mask = _foreground_mask(image)
    components = _connected_components(mask)
    boxes = _merge_detached_dots(components)
    boxes = _split_at_clear_valleys(mask, boxes)
    return mask, tuple(sorted(boxes, key=lambda box: (box[0], box[1])))


def segment_line(image: np.ndarray) -> tuple[tuple[int, int, int, int], ...]:
    """Tìm bounding boxes ký tự, theo tọa độ ảnh và thứ tự trái sang phải."""
    _, boxes = _analyze_line(image)
    return boxes


def _padded_crop(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    padding = max(2, round(max(x1 - x0, y1 - y0) * 0.12))
    return image[
        max(0, y0 - padding) : min(image.shape[0], y1 + padding),
        max(0, x0 - padding) : min(image.shape[1], x1 + padding),
    ]


def crop_line_regions(
    image: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[np.ndarray, ...]:
    """Lấy các vùng ký tự đã thêm padding để hiển thị hoặc kiểm tra."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Ảnh dòng phải là mảng grayscale 2D không rỗng.")
    return tuple(_padded_crop(image, box) for box in boxes)


def save_segmentation_debug(
    image_path: Path,
    output_dir: Path,
    result: LineRecognitionResult | None = None,
) -> tuple[Path, ...]:
    """Lưu mask, ảnh gắn nhãn vùng và crop ký tự mà không ghi đè debug cũ."""
    grayscale = load_grayscale_image(image_path)
    mask, boxes = _analyze_line(grayscale)
    debug_dir = output_dir
    suffix = 1
    while debug_dir.exists():
        debug_dir = output_dir.with_name(f"{output_dir.name}_{suffix:02d}")
        suffix += 1
    debug_dir.mkdir(parents=True)

    mask_path = debug_dir / "01_binary_mask.png"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

    annotated = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for index, box in enumerate(boxes, start=1):
        draw.rectangle(box, outline=(255, 0, 0), width=2)
        draw.text((box[0], max(0, box[1] - 14)), str(index), fill=(255, 0, 0))
    annotated_path = debug_dir / "02_regions_numbered.png"
    annotated.save(annotated_path)

    crop_paths: list[Path] = []
    for index, crop in enumerate(crop_line_regions(grayscale, boxes), start=1):
        crop_path = debug_dir / f"crop_{index:02d}.png"
        Image.fromarray(crop, mode="L").save(crop_path)
        crop_paths.append(crop_path)

    if result is not None:
        if len(result.boxes) != len(boxes) or len(result.region_predictions) != len(boxes):
            raise ValueError("Kết quả nhận diện không khớp với các vùng debug.")
        with (debug_dir / "03_region_predictions.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                (
                    "region_index",
                    "x0",
                    "y0",
                    "x1",
                    "y1",
                    "width",
                    "height",
                    "predicted_character",
                )
            )
            for index, (box, prediction) in enumerate(
                zip(boxes, result.region_predictions), start=1
            ):
                x0, y0, x1, y1 = box
                writer.writerow(
                    (index, x0, y0, x1, y1, x1 - x0, y1 - y0, prediction)
                )
    return (mask_path, annotated_path, *crop_paths)


def recognize_line(
    image_path: Path,
    model: ManualKNN,
    k: int,
    character_mapping: dict[int, str],
) -> LineRecognitionResult:
    """Tách dòng, phân loại từng crop và ghép nhãn thành chuỗi."""
    grayscale = load_grayscale_image(image_path)
    _, boxes = _analyze_line(grayscale)
    if not boxes:
        return LineRecognitionResult(text="", boxes=())

    predictions: list[str] = []
    region_predictions: list[str] = []
    median_height = float(np.median([y1 - y0 for _, y0, _, y1 in boxes]))
    previous_right: int | None = None
    for box in boxes:
        x0, _, x1, _ = box
        if previous_right is not None:
            gap = x0 - previous_right
            if gap > max(4, median_height * 0.75):
                predictions.append(" ")
        crop = _padded_crop(grayscale, box)
        _, vector = preprocess_character_image_array(crop)
        class_id, _ = model.predict_one(vector, k)
        if class_id not in character_mapping:
            raise ValueError(f"Không có ký tự cho class ID dự đoán {class_id}.")
        prediction = character_mapping[class_id]
        predictions.append(prediction)
        region_predictions.append(prediction)
        previous_right = x1
    return LineRecognitionResult(
        text="".join(predictions),
        boxes=boxes,
        region_predictions=tuple(region_predictions),
    )
