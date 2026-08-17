from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import RegionOrder


logger = logging.getLogger(__name__)


@dataclass
class Region:
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    mean_lab: np.ndarray
    quality: float
    saliency: float = 0.0


@dataclass
class RegionPlan:
    regions: list[Region]
    residual: np.ndarray
    analysis_width: int
    analysis_height: int


class SegmentationUnavailable(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        return 0.0, 0.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _clean_mask(mask: np.ndarray, min_component: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(first & second))
    if intersection == 0:
        return 0.0, 0.0
    union = int(np.count_nonzero(first | second))
    smaller = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    return intersection / max(union, 1), intersection / max(smaller, 1)


def _make_region(mask: np.ndarray, lab: np.ndarray, quality: float) -> Region:
    area = int(np.count_nonzero(mask))
    mean_lab = lab[mask].mean(axis=0) if area else np.zeros(3, dtype=np.float32)
    return Region(
        mask=mask,
        area=area,
        bbox=_bbox(mask),
        centroid=_centroid(mask),
        mean_lab=np.asarray(mean_lab, dtype=np.float32),
        quality=float(quality),
    )


def _bbox_gap(first: Region, second: Region) -> tuple[float, float]:
    ax0, ay0, ax1, ay1 = first.bbox
    bx0, by0, bx1, by1 = second.bbox
    gap_x = max(0, max(ax0, bx0) - min(ax1, bx1))
    gap_y = max(0, max(ay0, by0) - min(ay1, by1))
    return float(gap_x), float(gap_y)


def _group_related(regions: list[Region], lab: np.ndarray) -> list[Region]:
    if len(regions) < 2:
        return regions
    height, width = lab.shape[:2]
    total = height * width
    diagonal = math.hypot(width, height)
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first_index, first in enumerate(regions):
        if first.area / total > 0.06:
            continue
        for second_index in range(first_index + 1, len(regions)):
            second = regions[second_index]
            if second.area / total > 0.06:
                continue
            color_distance = float(np.linalg.norm(first.mean_lab - second.mean_lab))
            if color_distance > 28.0:
                continue
            gap_x, gap_y = _bbox_gap(first, second)
            ax0, ay0, ax1, ay1 = first.bbox
            bx0, by0, bx1, by1 = second.bbox
            aligned_y = abs(first.centroid[1] - second.centroid[1]) < 0.7 * max(
                ay1 - ay0, by1 - by0, 1
            )
            aligned_x = abs(first.centroid[0] - second.centroid[0]) < 0.7 * max(
                ax1 - ax0, bx1 - bx0, 1
            )
            close = math.hypot(gap_x, gap_y) / max(diagonal, 1) < 0.075
            if close and (aligned_x or aligned_y):
                union(first_index, second_index)

    groups: dict[int, list[Region]] = {}
    for index, region in enumerate(regions):
        groups.setdefault(find(index), []).append(region)

    merged: list[Region] = []
    for group in groups.values():
        mask = np.zeros((height, width), dtype=bool)
        weighted_quality = 0.0
        total_area = 0
        for region in group:
            mask |= region.mask
            weighted_quality += region.quality * region.area
            total_area += region.area
        merged.append(_make_region(mask, lab, weighted_quality / max(total_area, 1)))
    return merged


def _score_regions(regions: list[Region], rgb: np.ndarray) -> None:
    height, width = rgb.shape[:2]
    total = height * width
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    gradient /= max(float(np.percentile(gradient, 98.0)), 1e-6)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    kernel = np.ones((5, 5), np.uint8)

    for region in regions:
        mask_u8 = region.mask.astype(np.uint8)
        ring = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool) & ~region.mask
        inside_lab = region.mean_lab
        outside_lab = lab[ring].mean(axis=0) if np.any(ring) else inside_lab
        contrast = min(float(np.linalg.norm(inside_lab - outside_lab)) / 90.0, 1.0)
        detail = min(float(gradient[region.mask].mean()), 1.0)
        colorful = float(saturation[region.mask].mean())
        nx = (region.centroid[0] / max(width, 1)) - 0.5
        ny = (region.centroid[1] / max(height, 1)) - 0.5
        center = max(0.0, 1.0 - math.hypot(nx, ny) / 0.71)
        area_ratio = region.area / max(total, 1)
        area_score = math.exp(-((math.log(max(area_ratio, 1e-5)) - math.log(0.08)) / 1.5) ** 2)
        region.saliency = (
            0.28 * contrast
            + 0.24 * detail
            + 0.20 * center
            + 0.13 * colorful
            + 0.10 * area_score
            + 0.05 * region.quality
        )


def _order_regions(
    regions: list[Region],
    order: RegionOrder,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> list[Region]:
    if order == RegionOrder.reading_order:
        return sorted(regions, key=lambda item: (item.centroid[1] / height, item.centroid[0] / width))
    if order == RegionOrder.center_first:
        return sorted(
            regions,
            key=lambda item: math.hypot(item.centroid[0] / width - 0.5, item.centroid[1] / height - 0.5),
        )
    if order == RegionOrder.large_first:
        return sorted(regions, key=lambda item: item.area, reverse=True)
    if order == RegionOrder.small_first:
        return sorted(regions, key=lambda item: item.area)
    if order == RegionOrder.random:
        shuffled = list(regions)
        rng.shuffle(shuffled)
        return shuffled
    return sorted(regions, key=lambda item: item.saliency, reverse=True)


def postprocess_masks(
    candidates: list[dict[str, Any]],
    rgb: np.ndarray,
    max_regions: int,
    min_area_ratio: float,
    order: RegionOrder,
    seed: int,
) -> RegionPlan:
    height, width = rgb.shape[:2]
    total = height * width
    min_area = max(16, round(total * min_area_ratio))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    prepared: list[Region] = []
    for candidate in candidates:
        mask = np.asarray(candidate.get("segmentation"), dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = _clean_mask(mask, max(8, min_area // 5))
        area = int(np.count_nonzero(mask))
        if area < min_area or area / total > 0.94:
            continue
        quality = float(candidate.get("predicted_iou", 0.5)) * float(
            candidate.get("stability_score", 1.0)
        )
        prepared.append(_make_region(mask, lab, quality))

    prepared.sort(key=lambda item: (item.quality, -item.area), reverse=True)
    deduplicated: list[Region] = []
    for candidate in prepared:
        duplicate = False
        for selected in deduplicated:
            iou, containment = _mask_iou(candidate.mask, selected.mask)
            area_ratio = min(candidate.area, selected.area) / max(candidate.area, selected.area, 1)
            if iou > 0.78 or (containment > 0.96 and area_ratio > 0.72):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(candidate)
        if len(deduplicated) >= max_regions * 4:
            break

    assigned = np.zeros((height, width), dtype=bool)
    partitioned: list[Region] = []
    for candidate in sorted(deduplicated, key=lambda item: (item.area, -item.quality)):
        unique = candidate.mask & ~assigned
        if int(np.count_nonzero(unique)) < min_area:
            continue
        region = _make_region(unique, lab, candidate.quality)
        partitioned.append(region)
        assigned |= unique

    grouped = _group_related(partitioned, lab)
    _score_regions(grouped, rgb)
    rng = np.random.default_rng(seed)
    ordered = _order_regions(grouped, order, width, height, rng)[:max_regions]

    covered = np.zeros((height, width), dtype=bool)
    final_regions: list[Region] = []
    for region in ordered:
        unique = region.mask & ~covered
        if int(np.count_nonzero(unique)) < min_area:
            continue
        final = _make_region(unique, lab, region.quality)
        final.saliency = region.saliency
        final_regions.append(final)
        covered |= unique

    return RegionPlan(
        regions=final_regions,
        residual=~covered,
        analysis_width=width,
        analysis_height=height,
    )


class Sam2Segmenter:
    def __init__(self) -> None:
        self.checkpoint = Path(
            os.getenv("SAM2_CHECKPOINT", "/app/models/sam2.1_hiera_tiny.pt")
        )
        self.config = os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.device_request = os.getenv("SAM2_DEVICE", "auto").lower()
        self.points_per_side = _env_int("SAM2_POINTS_PER_SIDE", 16)
        self.points_per_batch = _env_int("SAM2_POINTS_PER_BATCH", 16)
        self.analysis_size = _env_int("SAM2_ANALYSIS_SIZE", 768)
        self._generator: Any | None = None
        self._torch: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> None:
        if self._generator is not None:
            return
        with self._load_lock:
            if self._generator is not None:
                return
            try:
                import torch
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.build_sam import build_sam2
            except ImportError as exc:
                raise SegmentationUnavailable(
                    "SAM 2 no está instalado; usa segmentation_mode=none o la imagen Docker 0.3"
                ) from exc
            if not self.checkpoint.is_file():
                raise SegmentationUnavailable(f"No existe el checkpoint SAM 2: {self.checkpoint}")
            if self.device_request == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.device_request
            if device == "cpu":
                torch.set_num_threads(_env_int("SAM2_CPU_THREADS", 4))
            logger.info("Loading SAM 2 on %s", device)
            model = build_sam2(
                self.config,
                str(self.checkpoint),
                device=device,
                apply_postprocessing=False,
            )
            self._generator = SAM2AutomaticMaskGenerator(
                model=model,
                points_per_side=self.points_per_side,
                points_per_batch=self.points_per_batch,
                pred_iou_thresh=0.72,
                stability_score_thresh=0.86,
                box_nms_thresh=0.72,
                crop_n_layers=0,
                min_mask_region_area=0,
                output_mode="binary_mask",
            )
            self._torch = torch

    def generate(
        self,
        rgb: np.ndarray,
        max_regions: int,
        min_area_ratio: float,
        order: RegionOrder,
        seed: int,
    ) -> RegionPlan:
        self._load()
        height, width = rgb.shape[:2]
        scale = min(1.0, self.analysis_size / max(height, width))
        if scale < 1.0:
            analysis = cv2.resize(
                rgb,
                (max(2, round(width * scale)), max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            analysis = rgb
        with self._inference_lock:
            assert self._torch is not None and self._generator is not None
            with self._torch.inference_mode():
                candidates = self._generator.generate(analysis)
        plan = postprocess_masks(
            candidates,
            analysis,
            max_regions=max_regions,
            min_area_ratio=min_area_ratio,
            order=order,
            seed=seed,
        )
        if analysis.shape[:2] == (height, width):
            return plan

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        scaled_regions: list[Region] = []
        for region in plan.regions:
            mask = cv2.resize(
                region.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            scaled = _make_region(mask, lab, region.quality)
            scaled.saliency = region.saliency
            scaled_regions.append(scaled)
        covered = np.zeros((height, width), dtype=bool)
        for region in scaled_regions:
            region.mask &= ~covered
            covered |= region.mask
        return RegionPlan(
            regions=scaled_regions,
            residual=~covered,
            analysis_width=analysis.shape[1],
            analysis_height=analysis.shape[0],
        )


_segmenter = Sam2Segmenter()


def segment_image(
    rgb: np.ndarray,
    max_regions: int,
    min_area_ratio: float,
    order: RegionOrder,
    seed: int,
) -> RegionPlan:
    return _segmenter.generate(rgb, max_regions, min_area_ratio, order, seed)


def save_region_preview(rgb: np.ndarray, plan: RegionPlan, path: Path) -> None:
    preview = rgb.astype(np.float32)
    palette = [
        (236, 72, 72),
        (59, 130, 246),
        (34, 197, 94),
        (250, 204, 21),
        (168, 85, 247),
        (249, 115, 22),
        (20, 184, 166),
        (244, 114, 182),
    ]
    for index, region in enumerate(plan.regions):
        color = np.array(palette[index % len(palette)], dtype=np.float32)
        preview[region.mask] = preview[region.mask] * 0.62 + color * 0.38
        contours, _ = cv2.findContours(
            region.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, color.tolist(), 2)
        x, y, _, _ = region.bbox
        cv2.putText(
            preview,
            str(index + 1),
            (x + 4, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            str(index + 1),
            (x + 4, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(preview, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
