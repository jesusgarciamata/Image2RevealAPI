from __future__ import annotations

import numpy as np

from app.models import Direction, RegionOrder
from app.renderer import build_segmented_fill_arrival
from app.segmentation import RegionPlan, postprocess_masks


def test_postprocess_deduplicates_partitions_and_orders_masks() -> None:
    rgb = np.full((120, 180, 3), 235, dtype=np.uint8)
    rgb[10:35, 20:150] = [210, 35, 35]
    rgb[45:110, 65:115] = [25, 80, 180]
    title = np.zeros((120, 180), dtype=bool)
    title[10:35, 20:150] = True
    duplicate_title = title.copy()
    person = np.zeros_like(title)
    person[45:110, 65:115] = True
    broad = np.zeros_like(title)
    broad[5:115, 5:175] = True

    plan = postprocess_masks(
        [
            {"segmentation": title, "predicted_iou": 0.96, "stability_score": 0.98},
            {"segmentation": duplicate_title, "predicted_iou": 0.91, "stability_score": 0.95},
            {"segmentation": person, "predicted_iou": 0.97, "stability_score": 0.98},
            {"segmentation": broad, "predicted_iou": 0.80, "stability_score": 0.90},
        ],
        rgb,
        max_regions=8,
        min_area_ratio=0.002,
        order=RegionOrder.reading_order,
        seed=1,
    )

    assert len(plan.regions) >= 2
    coverage = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for region in plan.regions:
        coverage += region.mask.astype(np.uint8)
    assert int(coverage.max()) == 1
    assert np.all((coverage.astype(bool) | plan.residual))


def test_nested_object_is_not_discarded_as_duplicate() -> None:
    rgb = np.full((100, 160, 3), [35, 35, 35], dtype=np.uint8)
    background = np.zeros((100, 160), dtype=bool)
    background[5:95, 5:155] = True
    subject = np.zeros_like(background)
    subject[25:90, 55:105] = True
    rgb[subject] = [220, 45, 45]
    plan = postprocess_masks(
        [
            {"segmentation": background, "predicted_iou": 0.97, "stability_score": 0.98},
            {"segmentation": subject, "predicted_iou": 0.96, "stability_score": 0.98},
        ],
        rgb,
        max_regions=6,
        min_area_ratio=0.002,
        order=RegionOrder.center_first,
        seed=1,
    )
    assert any(np.count_nonzero(region.mask & subject) > subject.sum() * 0.9 for region in plan.regions)


def test_segmented_arrival_finishes_regions_sequentially() -> None:
    rgb = np.full((80, 140, 3), 255, dtype=np.uint8)
    first = np.zeros((80, 140), dtype=bool)
    first[10:35, 10:60] = True
    second = np.zeros_like(first)
    second[40:72, 75:130] = True
    plan = postprocess_masks(
        [
            {"segmentation": first, "predicted_iou": 0.99, "stability_score": 0.99},
            {"segmentation": second, "predicted_iou": 0.98, "stability_score": 0.99},
        ],
        rgb,
        max_regions=4,
        min_area_ratio=0.001,
        order=RegionOrder.reading_order,
        seed=3,
    )
    arrival = build_segmented_fill_arrival(
        plan,
        radius_ratio=0.12,
        brush_count=2,
        direction=Direction.reading_order,
        rng=np.random.default_rng(3),
    )
    assert arrival.shape == first.shape
    assert np.isfinite(arrival).all()
    assert float(arrival[plan.regions[0].mask].max()) <= float(
        arrival[plan.regions[1].mask].min()
    ) + 1e-5
