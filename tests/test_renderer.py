from __future__ import annotations

import numpy as np

from app.models import DetailMode, Direction, RenderOptions
from app.renderer import (
    build_continuous_brush_path,
    build_detail_arrival,
    build_detail_mask,
    build_detail_source,
    build_fill_arrival,
    build_random_residual_arrival,
    build_random_residual_brush_path,
    harden_detail_mask,
)


def test_detail_mask_finds_colored_and_dark_edges() -> None:
    image = np.full((100, 160, 3), 240, dtype=np.uint8)
    image[20:80, 30:34] = [20, 20, 20]
    image[35:65, 80:120] = [200, 80, 20]
    mask = build_detail_mask(image)
    assert mask.shape == image.shape[:2]
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0
    assert float(mask[:, 30:36].mean()) > float(mask[:, 0:10].mean())
    assert float(mask[33:67, 78:122].mean()) > float(mask[0:10, 0:10].mean())


def test_detail_selectivity_reduces_weak_texture() -> None:
    rng = np.random.default_rng(12)
    image = np.full((120, 180, 3), 225, dtype=np.int16)
    texture = rng.integers(-14, 15, size=image.shape[:2], dtype=np.int16)
    image += texture[..., None]
    image[20:100, 88:92] = [15, 15, 15]
    image = np.clip(image, 0, 255).astype(np.uint8)

    permissive = build_detail_mask(image, 0.2)
    selective = build_detail_mask(image, 0.8)

    weak_texture = np.ones(image.shape[:2], dtype=bool)
    weak_texture[:, 82:98] = False
    assert float(selective[weak_texture].mean()) < float(permissive[weak_texture].mean())
    assert float(selective[:, 86:94].mean()) > float(selective[weak_texture].mean())


def test_detail_chroma_controls_only_color() -> None:
    image = np.array([[[210, 80, 20], [20, 160, 220]]], dtype=np.uint8)

    monochrome = build_detail_source(image, 0.0)
    muted = build_detail_source(image, 0.2)
    original = build_detail_source(image, 1.0)

    np.testing.assert_array_equal(monochrome[..., 0], monochrome[..., 1])
    np.testing.assert_array_equal(monochrome[..., 1], monochrome[..., 2])
    np.testing.assert_array_equal(original, image.astype(np.float32))
    assert np.abs(muted[..., 0] - muted[..., 2]).mean() < np.abs(
        original[..., 0] - original[..., 2]
    ).mean()


def test_region_detail_mode_is_the_new_default() -> None:
    options = RenderOptions()
    assert options.detail_mode == DetailMode.regions
    assert options.detail_feather == 0.006


def test_harden_detail_mask_suppresses_haze_and_keeps_ink() -> None:
    mask = np.array([[0.0, 0.10, 0.30, 0.60, 0.90, 1.0]], dtype=np.float32)
    hardened = harden_detail_mask(mask)

    assert hardened.dtype == np.float32
    assert hardened[0, 1] < mask[0, 1]
    assert hardened[0, 4] > mask[0, 4]
    assert hardened[0, 0] == 0.0
    assert hardened[0, -1] == 1.0


def test_arrival_maps_cover_full_image() -> None:
    rng = np.random.default_rng(100)
    detail = build_detail_arrival(90, 160, Direction.right_to_left, rng)
    fill = build_fill_arrival(90, 160, 0.12, 3, Direction.reading_order, rng)
    for arrival in (detail, fill):
        assert arrival.shape == (90, 160)
        assert np.isfinite(arrival).all()
        assert 0.0 <= float(arrival.min())
        assert float(arrival.max()) <= 1.0


def test_seed_is_reproducible() -> None:
    first = build_detail_arrival(
        50, 80, Direction.organic, np.random.default_rng(99)
    )
    second = build_detail_arrival(
        50, 80, Direction.organic, np.random.default_rng(99)
    )
    np.testing.assert_array_equal(first, second)


def test_fill_uses_dense_continuous_paths() -> None:
    radius = 24.0
    for direction in Direction:
        path = build_continuous_brush_path(
            180, 320, radius, direction, np.random.default_rng(8)
        )
        steps = [
            np.hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(path, path[1:])
        ]
        assert len(path) > 100
        assert max(steps) < radius * 0.45, direction


def test_residual_path_is_dense_random_and_reproducible() -> None:
    residual = np.ones((120, 200), dtype=bool)
    residual[35:85, 70:130] = False
    radius = 18.0
    first = build_random_residual_brush_path(
        residual, radius, np.random.default_rng(42)
    )
    second = build_random_residual_brush_path(
        residual, radius, np.random.default_rng(42)
    )

    np.testing.assert_array_equal(first, second)
    steps = np.diff(np.asarray(first), axis=0)
    distances = np.linalg.norm(steps, axis=1)
    assert float(distances.max()) <= max(1.0, radius * 0.08) * 1.05
    assert np.any(steps[:, 0] > 0) and np.any(steps[:, 0] < 0)
    assert np.any(steps[:, 1] > 0) and np.any(steps[:, 1] < 0)


def test_random_residual_arrival_covers_only_residual() -> None:
    residual = np.zeros((90, 150), dtype=bool)
    residual[5:42, 8:140] = True
    residual[55:85, 25:120] = True
    arrival = build_random_residual_arrival(
        residual,
        radius_ratio=0.08,
        brush_count=2,
        rng=np.random.default_rng(7),
    )

    assert np.isfinite(arrival).all()
    assert 0.0 <= float(arrival[residual].min())
    assert float(arrival[residual].max()) <= 1.0
    assert np.all(arrival[~residual] == 1.0)
    horizontal = np.abs(np.diff(arrival, axis=1))[residual[:, :-1] & residual[:, 1:]]
    vertical = np.abs(np.diff(arrival, axis=0))[residual[:-1] & residual[1:]]
    neighbor_deltas = np.concatenate((horizontal, vertical))
    assert float(np.percentile(neighbor_deltas, 99)) < 0.08
