from __future__ import annotations

import numpy as np

from app.models import Direction
from app.renderer import build_detail_arrival, build_detail_mask, build_fill_arrival


def test_detail_mask_finds_colored_and_dark_edges() -> None:
    image = np.full((100, 160, 3), 240, dtype=np.uint8)
    image[20:80, 30:34] = [20, 20, 20]
    image[35:65, 80:120] = [200, 80, 20]
    mask = build_detail_mask(image)
    assert mask.shape == image.shape[:2]
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0
    assert float(mask[:, 30:36].mean()) > float(mask[:, 0:10].mean())
    assert float(mask[33:67, 78:122].mean()) > float(mask[0:10, 0:10].mean())


def test_arrival_maps_cover_full_image() -> None:
    rng = np.random.default_rng(100)
    detail = build_detail_arrival(90, 160, Direction.right_to_left, rng)
    fill = build_fill_arrival(90, 160, 0.12, Direction.right_to_left, rng)
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

