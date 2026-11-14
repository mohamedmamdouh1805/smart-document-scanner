from itertools import permutations

import cv2
import numpy as np
import pytest

from src.image_utils import InvalidImageError, encode_png
from src.scanner import (
    DocumentNotFoundError,
    detect_document,
    four_point_transform,
    order_points,
    render_scan,
    rotate_scan,
    scan_document,
    validate_manual_corners,
)


def test_orders_every_permutation_of_a_skewed_quadrilateral():
    expected = np.array([[50, 40], [260, 75], [320, 200], [90, 170]], dtype=np.float32)
    for permuted in permutations(expected):
        np.testing.assert_array_equal(order_points(permuted), expected)


def test_orders_a_diamond_with_tied_coordinate_sums():
    expected = np.array([[80, 20], [140, 80], [80, 140], [20, 80]], dtype=np.float32)
    np.testing.assert_array_equal(order_points(expected[[3, 1, 0, 2]]), expected)


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[0, 0], [0, 0], [10, 10], [0, 10]],
        [[0, 0], [10, 0], [0, 10], [2, 2]],
        [[0, 0], [10, 0], [10, np.nan], [0, 10]],
    ],
)
def test_rejects_invalid_corners(points):
    with pytest.raises(ValueError):
        order_points(points)


def test_perspective_transform_preserves_corner_positions_and_output_dimensions():
    image = np.zeros((260, 360, 3), dtype=np.uint8)
    points = np.array([[40, 30], [280, 30], [245, 190], [40, 190]], dtype=np.float32)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    for point, color in zip(points.astype(int), colors):
        cv2.circle(image, tuple(point), 8, color, -1)

    result = four_point_transform(image, points[[2, 0, 3, 1]])

    assert result.shape == (164, 240, 3)
    np.testing.assert_array_equal(
        [result[0, 0], result[0, -1], result[-1, -1], result[-1, 0]], colors
    )


@pytest.mark.parametrize(
    ("shape", "points", "portrait"),
    [
        ((680, 500), [[100, 60], [390, 90], [415, 590], [70, 600]], True),
        ((500, 680), [[85, 70], [590, 95], [610, 405], [55, 390]], False),
    ],
)
@pytest.mark.parametrize("scale", [0.5, 1, 3])
def test_detects_and_scans_document_at_multiple_resolutions(shape, points, portrait, scale):
    height, width = (round(size * scale) for size in shape)
    corners = np.rint(np.array(points) * scale).astype(np.int32)
    photo = np.full((height, width, 3), 35, dtype=np.uint8)
    cv2.fillConvexPoly(photo, corners, (224, 235, 244))
    cv2.putText(
        photo, "Report", (round(135 * scale), round(180 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), max(1, round(2 * scale)),
    )

    np.testing.assert_allclose(detect_document(photo), corners, atol=max(3, 4 * scale))
    result = scan_document(photo)
    scan_height, scan_width = result.shape[:2]

    assert (scan_height > scan_width) == portrait
    assert scan_height < height and scan_width < width
    assert np.all(result[-10, scan_width // 2] > 200)


@pytest.mark.parametrize("has_circle", [False, True])
def test_reports_detection_failure_without_full_image_fallback(has_circle):
    blank = np.full((400, 400, 3), 75, dtype=np.uint8)
    if has_circle:
        cv2.circle(blank, (200, 200), 130, (240, 240, 240), -1)
    with pytest.raises(DocumentNotFoundError, match="No document boundary"):
        scan_document(blank)


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.empty((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_rejects_invalid_image_arrays(image):
    with pytest.raises(InvalidImageError):
        scan_document(image)


def test_scan_modes_preserve_dimensions_and_render_dark_marks():
    image = np.full((180, 260, 3), (205, 218, 230), dtype=np.uint8)
    cv2.line(image, (30, 80), (230, 80), (35, 35, 35), 3)

    color = render_scan(image, "color")
    gray = render_scan(image, "grayscale")
    binary = render_scan(image, "bw")

    np.testing.assert_array_equal(color, image)
    assert gray.shape == binary.shape == (180, 260)
    assert gray.dtype == binary.dtype == np.uint8
    assert gray[80, 125] < gray[100, 125]
    assert set(np.unique(binary)) == {0, 255}
    assert binary[80, 125] == 0
    assert binary[100, 125] == 255


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown scan mode"):
        render_scan(np.zeros((10, 10, 3), dtype=np.uint8), "other")


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, [[1, 2, 3], [4, 5, 6]]),
        (90, [[4, 1], [5, 2], [6, 3]]),
        (180, [[6, 5, 4], [3, 2, 1]]),
        (270, [[3, 6], [2, 5], [1, 4]]),
    ],
)
@pytest.mark.parametrize("color", [False, True])
def test_rotate_scan_maps_pixels_clockwise(rotation, expected, color):
    scan = np.arange(1, 7, dtype=np.uint8).reshape(2, 3)
    expected = np.array(expected, dtype=np.uint8)
    if color:
        scan = np.stack([scan, scan + 10, scan + 20], axis=-1)
        expected = np.stack([expected, expected + 10, expected + 20], axis=-1)
    np.testing.assert_array_equal(rotate_scan(scan, rotation), expected)


def test_rejects_unsupported_rotation():
    with pytest.raises(ValueError, match="degrees clockwise"):
        rotate_scan(np.zeros((5, 8), dtype=np.uint8), 45)


def test_manual_corners_accept_shuffled_points_on_image_edges():
    expected = [[0, 0], [39, 0], [39, 29], [0, 29]]
    points = [expected[index] for index in (2, 0, 3, 1)]
    np.testing.assert_array_equal(validate_manual_corners(points, (30, 40, 3)), expected)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([[0, 0], [20, 0], [20, 0], [0, 20]], "distinct"),
        ([[0, 0], [20, 0], [15, 0], [5, 0]], "convex"),
        ([[0, 0], [20, 0], [8, 8], [0, 20]], "convex"),
        ([[0, 0], [np.inf, 0], [20, 20], [0, 20]], "convex"),
        ([[-1, 0], [20, 0], [20, 20], [0, 20]], "within the photo"),
        ([[0, 0], [40, 0], [40, 20], [0, 20]], "within the photo"),
        ([[0, 0], [20, 0], [20, 30], [0, 30]], "within the photo"),
        ([[0, 0], [2, 0], [2, 2], [0, 2]], "too small"),
    ],
)
def test_rejects_invalid_manual_corners(points, message):
    with pytest.raises(InvalidImageError, match=message):
        validate_manual_corners(points, (30, 40, 3))


@pytest.mark.parametrize("mode", ["color", "grayscale", "bw"])
def test_manual_scan_warps_and_rotates_before_png_export(mode):
    photo = np.full((70, 110, 3), 220, dtype=np.uint8)
    points = np.array([[12, 10], [89, 12], [81, 58], [7, 55]], dtype=np.float32)
    cv2.line(photo, (22, 30), (69, 35), (32, 32, 32), 2)
    with pytest.raises(DocumentNotFoundError):
        scan_document(photo)

    corrected = four_point_transform(photo, points)
    result = scan_document(photo, mode, corners=points[[2, 0, 3, 1]], rotation=90)
    exported = cv2.imdecode(np.frombuffer(encode_png(result), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    assert result.shape[:2] == (77, 47)
    expected = np.rot90(render_scan(corrected, mode), k=-1)
    np.testing.assert_array_equal(exported, expected)


def test_automatic_scan_still_supports_output_rotation():
    photo = np.full((300, 400, 3), 30, dtype=np.uint8)
    cv2.rectangle(photo, (70, 65), (330, 235), (220, 230, 240), -1)

    original = scan_document(photo)
    rotated = scan_document(photo, rotation=270)

    np.testing.assert_array_equal(rotated, np.rot90(original))
