from __future__ import annotations

import numpy as np

from app.models import Direction
from app.renderer import (
    build_continuous_brush_path,
    build_detail_arrival,
    build_detail_mask,
    build_detail_source,
    build_fill_arrival,
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
