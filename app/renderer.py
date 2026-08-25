from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .models import DetailMode, Direction, RenderOptions, SegmentationMode
from .segmentation import RegionPlan, save_region_preview, segment_image


ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class RenderResult:
    width: int
    height: int
    frames: int
    regions_detected: int


def _runtime_threads(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


FFMPEG_THREADS = _runtime_threads("FFMPEG_THREADS", 2)
cv2.setNumThreads(_runtime_threads("OPENCV_THREADS", 1))


def _smoothstep(edge0: float, edge1: float, value: np.ndarray | float) -> np.ndarray:
    width = np.maximum(np.asarray(edge1) - np.asarray(edge0), 1e-6)
    scaled = np.clip((value - edge0) / width, 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _normalize_percentile(image: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    scale = float(np.percentile(image, percentile))
    if scale <= 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(image / scale, 0.0, 1.0).astype(np.float32)


def _hex_rgb(value: str) -> np.ndarray:
    return np.array([int(value[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.float32)


def _fit_image(image: np.ndarray, output_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if output_width and output_width != width:
        target_height = max(2, round(height * output_width / width))
        interpolation = cv2.INTER_AREA if output_width < width else cv2.INTER_CUBIC
        image = cv2.resize(image, (output_width, target_height), interpolation=interpolation)
    height, width = image.shape[:2]
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    return image[:even_height, :even_width]


def _paper_background(height: int, width: int, color: str) -> np.ndarray:
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:] = _hex_rgb(color)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx - width / 2) / max(width / 2, 1)
    ny = (yy - height / 2) / max(height / 2, 1)
    vignette = np.clip((nx * nx + ny * ny - 0.35) * 7.0, 0.0, 1.0)[..., None]
    return np.clip(base * (1.0 - 0.055 * vignette), 0, 255)


def build_detail_mask(rgb: np.ndarray, selectivity: float = 0.5) -> np.ndarray:
    """Create a soft mask for colored contours, hatching and high-frequency detail."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    luminance = cv2.GaussianBlur(gray, (0, 0), 0.75)
    gx = cv2.Scharr(luminance, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(luminance, cv2.CV_32F, 0, 1)
    luminance_edges = _normalize_percentile(cv2.magnitude(gx, gy), 97.5)

    color_edges = np.zeros_like(gray)
    for channel in cv2.split(lab):
        channel = cv2.GaussianBlur(channel, (0, 0), 0.9)
        cgx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        cgy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        color_edges = np.maximum(color_edges, cv2.magnitude(cgx, cgy))
    color_edges = _normalize_percentile(color_edges, 98.0)

    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9), normalize=True)
    local_square = cv2.boxFilter(gray * gray, cv2.CV_32F, (9, 9), normalize=True)
    local_variance = _normalize_percentile(
        np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0)), 97.0
    )

    fine = np.abs(
        cv2.GaussianBlur(gray, (0, 0), 0.7)
        - cv2.GaussianBlur(gray, (0, 0), 2.4)
    )
    fine = _normalize_percentile(fine, 97.0)
    darkness_support = np.clip((0.78 - gray) / 0.55, 0.0, 1.0) * local_variance

    feature = np.maximum(0.60 * luminance_edges, 0.48 * color_edges)
    feature = np.maximum(feature, 0.72 * local_variance)
    feature = np.maximum(feature, 0.78 * fine)
    feature = np.maximum(feature, 0.42 * darkness_support)
    # A selectivity of 0.5 reproduces the original 0.3.0 thresholds exactly.
    # Higher values keep only stronger contours; lower values retain more texture.
    lower_threshold = -0.10 + 0.40 * selectivity
    upper_threshold = 0.26 + 0.40 * selectivity
    mask = _smoothstep(lower_threshold, upper_threshold, feature)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.65)
    return np.clip(mask, 0.0, 1.0)


def build_detail_source(rgb: np.ndarray, chroma: float = 1.0) -> np.ndarray:
    """Return the source used by the detail phase with controllable chroma."""
    original = rgb.astype(np.float32)
    if chroma >= 1.0:
        return original
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None]
    return grayscale + (original - grayscale) * chroma


def harden_detail_mask(mask: np.ndarray) -> np.ndarray:
    """Turn the soft feature mask into defined ink while preserving antialiasing."""
    return _smoothstep(0.18, 0.72, mask).astype(np.float32)


def _low_frequency_noise(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    small_h = max(4, math.ceil(height / 100))
    small_w = max(4, math.ceil(width / 100))
    noise = rng.random((small_h, small_w), dtype=np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), max(height, width) / 100.0)
    noise -= noise.min()
    return noise / max(float(noise.max()), 1e-6)


def build_detail_arrival(
    height: int,
    width: int,
    direction: Direction,
    rng: np.random.Generator,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    if direction == Direction.reading_order:
        base = 0.58 * x + 0.42 * y
    elif direction == Direction.right_to_left:
        base = 1.0 - x
    elif direction == Direction.left_to_right:
        base = x
    elif direction == Direction.top_to_bottom:
        base = y
    elif direction == Direction.bottom_to_top:
        base = 1.0 - y
    elif direction == Direction.center_out:
        base = np.sqrt(((x - 0.5) / 0.71) ** 2 + ((y - 0.5) / 0.71) ** 2)
    elif direction == Direction.random_origins:
        origin_count = int(rng.integers(2, 5))
        base = np.full((height, width), np.inf, dtype=np.float32)
        for _ in range(origin_count):
            ox = float(rng.uniform(0.08, 0.92))
            oy = float(rng.uniform(0.08, 0.92))
            distance = np.sqrt(((x - ox) / 1.0) ** 2 + ((y - oy) / 1.0) ** 2)
            base = np.minimum(base, distance)
    else:
        base = 0.56 * (1.0 - x) + 0.22 * y + 0.22 * np.sin(y * math.pi * 3.0) ** 2

    noise = _low_frequency_noise(height, width, rng)
    arrival = base + (noise - 0.5) * 0.24
    arrival -= float(arrival.min())
    arrival /= max(float(arrival.max()), 1e-6)
    return arrival.astype(np.float32)


def _sample_line(
    start: tuple[float, float],
    end: tuple[float, float],
    step: float,
) -> list[tuple[float, float]]:
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    count = max(1, math.ceil(distance / max(step, 1.0)))
    return [
        (
            start[0] + (end[0] - start[0]) * index / count,
            start[1] + (end[1] - start[1]) * index / count,
        )
        for index in range(count)
    ]


def build_continuous_brush_path(
    height: int,
    width: int,
    radius: float,
    direction: Direction,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Build one dense, continuous serpentine path with a gently hand-driven wobble."""
    lane_spacing = radius * 1.28
    lane_count = max(2, math.ceil(height / lane_spacing))
    margin_y = min(radius * 0.16, height * 0.04)
    lanes = np.linspace(margin_y, height - 1 - margin_y, lane_count)
    if direction == Direction.bottom_to_top:
        lanes = lanes[::-1]

    start_on_left = direction != Direction.right_to_left
    step = max(2.0, radius * 0.24)
    margin_x = min(radius * 0.12, width * 0.03)
    path: list[tuple[float, float]] = []
    phase = float(rng.uniform(0.0, math.tau))

    for lane_index, lane_y in enumerate(lanes):
        left_to_right = (lane_index % 2 == 0) == start_on_left
        start_x, end_x = (
            (margin_x, width - 1 - margin_x)
            if left_to_right
            else (width - 1 - margin_x, margin_x)
        )
        segment = _sample_line((start_x, float(lane_y)), (end_x, float(lane_y)), step)
        for point_index, (x, y) in enumerate(segment):
            global_index = len(path) + point_index
            wobble = math.sin(global_index * 0.17 + phase) * radius * 0.13
            slow_wobble = math.sin(global_index * 0.043 + phase * 0.7) * radius * 0.11
            path.append((x, float(np.clip(y + wobble + slow_wobble, 0, height - 1))))

        if lane_index + 1 < len(lanes):
            connector_x = end_x
            connector = _sample_line(
                (connector_x, path[-1][1]),
                (connector_x, float(lanes[lane_index + 1])),
                step,
            )
            path.extend(connector)

    if direction == Direction.center_out:
        middle = len(path) // 2
        path = path[middle:] + path[-2::-1]
    elif direction in {Direction.organic, Direction.random_origins} and rng.random() < 0.5:
        path.reverse()
    return path


def build_fill_arrival(
    height: int,
    width: int,
    radius_ratio: float,
    brush_count: int,
    direction: Direction,
    rng: np.random.Generator,
) -> np.ndarray:
    radius = max(8.0, min(height, width) * radius_ratio)
    points = build_continuous_brush_path(height, width, radius, direction, rng)
    arrival = np.full((height, width), np.inf, dtype=np.float32)
    boundaries = np.linspace(0, len(points), brush_count + 1, dtype=int)

    for brush_index in range(brush_count):
        start = int(boundaries[brush_index])
        end = int(boundaries[brush_index + 1])
        brush_points = points[start:end]
        local_count = max(len(brush_points) - 1, 1)
        brush_phase = float(rng.uniform(0.0, math.tau))
        for local_index, (cx, cy) in enumerate(brush_points):
            scale = 1.0 + 0.08 * math.sin(local_index * 0.09 + brush_phase)
            point_radius = radius * scale
            stamp_time = local_index / local_count
            _stamp_arrival(arrival, cx, cy, point_radius, stamp_time, radius_ratio)

    finite = np.isfinite(arrival)
    if not finite.all():
        arrival[~finite] = 1.0
    arrival -= float(arrival.min())
    arrival /= max(float(arrival.max()), 1e-6)
    return arrival


def build_random_residual_brush_path(
    residual: np.ndarray,
    radius: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Visit residual cells in locally random directions with dense transitions."""
    height, width = residual.shape
    cell_size = max(2, round(radius * 0.55))
    rows = math.ceil(height / cell_size)
    columns = math.ceil(width / cell_size)
    active = np.zeros((rows, columns), dtype=bool)
    targets = np.zeros((rows, columns, 2), dtype=np.float32)

    for row in range(rows):
        y0 = row * cell_size
        y1 = min(height, y0 + cell_size)
        for column in range(columns):
            x0 = column * cell_size
            x1 = min(width, x0 + cell_size)
            positions = np.flatnonzero(residual[y0:y1, x0:x1])
            if not len(positions):
                continue
            chosen = int(positions[int(rng.integers(0, len(positions)))])
            local_width = x1 - x0
            targets[row, column] = (
                x0 + chosen % local_width,
                y0 + chosen // local_width,
            )
            active[row, column] = True

    remaining = int(np.count_nonzero(active))
    if not remaining:
        return []

    start_candidates = np.argwhere(active)
    start_row, start_column = start_candidates[int(rng.integers(0, len(start_candidates)))]
    current_row = int(start_row)
    current_column = int(start_column)
    ordered: list[tuple[float, float]] = []

    while remaining:
        if active[current_row, current_column]:
            active[current_row, current_column] = False
            remaining -= 1
            target = targets[current_row, current_column]
            ordered.append((float(target[0]), float(target[1])))
        if not remaining:
            break

        candidates: list[tuple[int, int]] = []
        for search_radius in range(1, 4):
            row0 = max(0, current_row - search_radius)
            row1 = min(rows, current_row + search_radius + 1)
            column0 = max(0, current_column - search_radius)
            column1 = min(columns, current_column + search_radius + 1)
            local = np.argwhere(active[row0:row1, column0:column1])
            if len(local):
                candidates = [
                    (int(row0 + item[0]), int(column0 + item[1]))
                    for item in local
                ]
                break
        if not candidates:
            global_candidates = np.argwhere(active)
            chosen = global_candidates[int(rng.integers(0, len(global_candidates)))]
            current_row, current_column = int(chosen[0]), int(chosen[1])
            continue

        distances = np.array(
            [
                math.hypot(row - current_row, column - current_column)
                for row, column in candidates
            ],
            dtype=np.float32,
        )
        weights = 1.0 / np.maximum(distances, 0.5)
        weights /= float(weights.sum())
        selected = int(rng.choice(len(candidates), p=weights))
        current_row, current_column = candidates[selected]

    dense: list[tuple[float, float]] = [ordered[0]]
    step = max(1.0, radius * 0.08)
    for target in ordered[1:]:
        segment = _sample_line(dense[-1], target, step)
        dense.extend(segment[1:])
        dense.append(target)
    return dense


def build_random_residual_arrival(
    residual: np.ndarray,
    radius_ratio: float,
    brush_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paint only unassigned pixels using dense, randomized local motion."""
    height, width = residual.shape
    radius = max(8.0, min(height, width) * radius_ratio)
    points = build_random_residual_brush_path(residual, radius, rng)
    arrival = np.full((height, width), np.inf, dtype=np.float32)
    if not points:
        return np.ones((height, width), dtype=np.float32)

    boundaries = np.linspace(0, len(points), brush_count + 1, dtype=int)
    for brush_index in range(brush_count):
        start = int(boundaries[brush_index])
        end = int(boundaries[brush_index + 1])
        brush_points = points[start:end]
        local_count = max(len(brush_points) - 1, 1)
        brush_phase = float(rng.uniform(0.0, math.tau))
        for local_index, (cx, cy) in enumerate(brush_points):
            scale = 1.0 + 0.035 * math.sin(local_index * 0.07 + brush_phase)
            _stamp_arrival(
                arrival,
                cx,
                cy,
                radius * scale,
                local_index / local_count,
                radius_ratio,
                radial_delay=0.0,
            )

    finite_residual = np.isfinite(arrival) & residual
    if np.any(finite_residual):
        lower = float(arrival[finite_residual].min())
        upper = float(arrival[finite_residual].max())
        arrival[finite_residual] = np.clip(
            (arrival[finite_residual] - lower) / max(upper - lower, 1e-6),
            0.0,
            1.0,
        )
    arrival[residual & ~finite_residual] = 1.0
    arrival[~residual] = 1.0
    return arrival


def _region_local_arrival(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    arrival = np.ones((height, width), dtype=np.float32)
    if not len(xs):
        return arrival

    sample_step = max(1, len(xs) // 80_000)
    points = np.column_stack((xs[::sample_step], ys[::sample_step])).astype(np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) >= 3:
        covariance = np.cov(centered, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))].astype(np.float32)
    else:
        axis = np.array([1.0, 0.0], dtype=np.float32)
    reading_vector = np.array([0.82, 0.38], dtype=np.float32)
    if float(np.dot(axis, reading_vector)) < 0:
        axis *= -1.0

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    projection = xx * axis[0] + yy * axis[1]
    values_inside = projection[mask]
    projection = (projection - float(values_inside.min())) / max(
        float(values_inside.max() - values_inside.min()), 1e-6
    )

    noise = _low_frequency_noise(height, width, rng)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    distance /= max(float(distance.max()), 1e-6)
    local = 0.70 * projection + 0.19 * noise + 0.11 * (1.0 - distance)
    local_values = local[mask]
    local = (local - float(local_values.min())) / max(
        float(local_values.max() - local_values.min()), 1e-6
    )
    arrival[mask] = np.clip(local[mask], 0.0, 1.0)
    return arrival


def build_segmented_fill_arrival(
    plan: RegionPlan,
    radius_ratio: float,
    brush_count: int,
    direction: Direction,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = plan.residual.shape
    arrival = np.ones((height, width), dtype=np.float32)
    weights = [max(0.65, math.sqrt(region.area / max(height * width, 1)) * 3.0) for region in plan.regions]
    residual_area = int(np.count_nonzero(plan.residual))
    residual_weight = max(0.9, min(1.8, math.sqrt(residual_area / max(height * width, 1)) * 2.0))
    total_weight = sum(weights) + residual_weight
    cursor = 0.0

    for region, weight in zip(plan.regions, weights):
        span = weight / max(total_weight, 1e-6)
        local = _region_local_arrival(region.mask, rng)
        arrival[region.mask] = cursor + local[region.mask] * span
        cursor += span

    if residual_area:
        residual_local = build_random_residual_arrival(
            plan.residual,
            radius_ratio,
            brush_count,
            rng,
        )
        residual_span = max(1.0 - cursor, 1e-6)
        arrival[plan.residual] = cursor + residual_local[plan.residual] * residual_span

    arrival -= float(arrival.min())
    arrival /= max(float(arrival.max()), 1e-6)
    return arrival


def _stamp_arrival(
    arrival: np.ndarray,
    cx: float,
    cy: float,
    point_radius: float,
    stamp_time: float,
    radius_ratio: float,
    radial_delay: float | None = None,
) -> None:
    height, width = arrival.shape
    x0 = max(0, int(cx - point_radius))
    x1 = min(width, int(cx + point_radius) + 1)
    y0 = max(0, int(cy - point_radius))
    y1 = min(height, int(cy + point_radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / point_radius
    inside = distance <= 1.0
    if radial_delay is None:
        radial_delay = 0.018 / max(radius_ratio, 0.03)
    candidate = stamp_time + distance.astype(np.float32) * radial_delay
    region = arrival[y0:y1, x0:x1]
    np.minimum(region, np.where(inside, candidate, np.inf), out=region)


def render_video(
    input_path: Path,
    output_path: Path,
    options: RenderOptions,
    progress: ProgressCallback,
) -> RenderResult:
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        rgb = np.asarray(image)

    rgb = _fit_image(rgb, options.output_width)
    height, width = rgb.shape[:2]
    total_frames = max(2, round(options.duration * options.fps))
    hold_frames = min(total_frames - 1, round(options.final_hold * options.fps))
    animation_frames = max(1, total_frames - hold_frames)

    rng = np.random.default_rng(options.seed)
    original = rgb.astype(np.float32)
    background = _paper_background(height, width, options.background)
    detail_alpha = build_detail_mask(rgb, options.detail_selectivity)
    detail_source = build_detail_source(rgb, options.detail_chroma)
    legacy_detail_arrival = build_detail_arrival(height, width, options.direction, rng)
    regions_detected = 0
    plan: RegionPlan | None = None
    if options.segmentation_mode == SegmentationMode.auto:
        plan = segment_image(
            rgb,
            max_regions=options.max_regions,
            min_area_ratio=options.min_region_area,
            order=options.region_order,
            seed=options.seed,
        )
        regions_detected = len(plan.regions)
        save_region_preview(rgb, plan, output_path.parent / "regions.png")

    if options.detail_mode == DetailMode.regions:
        detail_alpha = harden_detail_mask(detail_alpha)
        detail_rng = np.random.default_rng(options.seed ^ 0x05DE7A11)
        if plan is not None:
            detail_arrival = build_segmented_fill_arrival(
                plan,
                options.brush_radius,
                options.fill_brushes,
                options.direction,
                detail_rng,
            )
        else:
            detail_arrival = build_fill_arrival(
                height,
                width,
                options.brush_radius,
                options.fill_brushes,
                options.direction,
                detail_rng,
            )
        del legacy_detail_arrival
        detail_softness = options.detail_feather
    else:
        detail_arrival = legacy_detail_arrival
        detail_softness = 0.035

    if plan is not None:
        fill_arrival = build_segmented_fill_arrival(
            plan,
            options.brush_radius,
            options.fill_brushes,
            options.direction,
            rng,
        )
    else:
        fill_arrival = build_fill_arrival(
            height,
            width,
            options.brush_radius,
            options.fill_brushes,
            options.direction,
            rng,
        )
    progress(0.05)

    temporary = output_path.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(options.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-threads",
        str(FFMPEG_THREADS),
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("FFmpeg no abrió su entrada estándar")

    detail_end = options.detail_ratio
    fill_start = max(0.0, detail_end - options.fill_overlap)
    try:
        for frame_index in range(total_frames):
            if frame_index >= animation_frames:
                frame = original
            else:
                timeline = frame_index / max(animation_frames - 1, 1)
                detail_progress = float(
                    _smoothstep(0.0, detail_end, timeline)
                )
                detail_reveal = _smoothstep(
                    -detail_softness,
                    detail_softness,
                    detail_progress - detail_arrival,
                )
                detail = (detail_alpha * detail_reveal)[..., None]
                frame = background * (1.0 - detail) + detail_source * detail

                if timeline > fill_start:
                    fill_progress = float(
                        _smoothstep(fill_start, 1.0, timeline)
                    )
                    fill_softness = options.brush_feather * 0.22
                    fill = _smoothstep(
                        -fill_softness,
                        fill_softness,
                        fill_progress - fill_arrival,
                    )[..., None]
                    frame = frame * (1.0 - fill) + original * fill

                if timeline >= 0.999:
                    frame = original

            process.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
            progress(0.05 + 0.95 * (frame_index + 1) / total_frames)
    except (BrokenPipeError, OSError) as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"FFmpeg interrumpió el render: {stderr[-1000:]}") from exc
    finally:
        process.stdin.close()

    stderr_bytes = process.stderr.read() if process.stderr else b""
    return_code = process.wait()
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg terminó con código {return_code}: {stderr[-1000:]}")
    os.replace(temporary, output_path)
    return RenderResult(
        width=width,
        height=height,
        frames=total_frames,
        regions_detected=regions_detected,
    )
