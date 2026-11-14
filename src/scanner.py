"""Find a document contour, rectify its perspective, and render a scan."""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .image_utils import ImageArray, InvalidImageError

ScanMode = Literal["color", "grayscale", "bw"]
Rotation = Literal[0, 90, 180, 270]
Corners = NDArray[np.float32]
DETECTION_MAX_SIDE = 1200
MIN_DOCUMENT_AREA_RATIO = 0.10


class DocumentNotFoundError(ValueError):
    """No suitable four-sided document boundary was found."""


def _validate_image(image: ImageArray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 3
    ):
        raise InvalidImageError("Expected a uint8 BGR image at least 3 × 3 pixels.")


def order_points(points: ArrayLike) -> Corners:
    """Return four convex corners in top-left, top-right, bottom-right, bottom-left order."""
    corners = np.asarray(points, dtype=np.float32)
    if corners.shape not in ((4, 2), (4, 1, 2)) or not np.isfinite(corners).all():
        raise ValueError("Expected four finite (x, y) corner coordinates.")

    hull = cv2.convexHull(corners.reshape(4, 2), clockwise=False).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 1:
        raise ValueError("Corners must form a nondegenerate convex quadrilateral.")

    # The convex hull gives a consistent cyclic order, even when corners tie
    # on coordinate sums (for example, a diamond).
    x, y = hull.T
    start = np.lexsort((x, y, x + y))[0]
    return np.roll(hull, -start, axis=0)


def validate_manual_corners(points: ArrayLike, image_shape: tuple[int, ...]) -> Corners:
    """Validate four source-image pixel coordinates and return them in winding order."""
    try:
        corners = order_points(points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidImageError("Choose four distinct corners that form a convex quadrilateral.") from exc

    height, width = image_shape[:2]
    x, y = corners.T
    if np.any((x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)):
        raise InvalidImageError("All four corners must be within the photo.")

    edges = np.roll(corners, -1, axis=0) - corners
    if np.any(np.linalg.norm(edges, axis=1) < 3) or cv2.contourArea(corners) < 9:
        raise InvalidImageError("The selected document is too small to scan.")
    return corners


def detect_document(image: ImageArray) -> Corners:
    """Return document corners in source coordinates or raise DocumentNotFoundError."""
    _validate_image(image)
    height, width = image.shape[:2]
    scale = min(1.0, DETECTION_MAX_SIDE / max(height, width))
    if scale < 1:
        working_size = (max(3, round(width * scale)), max(3, round(height * scale)))
        working = cv2.resize(image, working_size, interpolation=cv2.INTER_AREA)
    else:
        working = image

    work_height, work_width = working.shape[:2]
    frame_area = work_width * work_height
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < MIN_DOCUMENT_AREA_RATIO * frame_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for tolerance in (0.015, 0.02, 0.03):
            polygon = cv2.approxPolyDP(contour, tolerance * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            polygon_area = cv2.contourArea(polygon)
            if polygon_area < MIN_DOCUMENT_AREA_RATIO * frame_area or area < 0.85 * polygon_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(polygon)
            if x <= 2 and y <= 2 and x + box_width >= work_width - 2 and y + box_height >= work_height - 2:
                continue
            corners = order_points(polygon)
            edges = np.roll(corners, -1, axis=0) - corners
            if np.min(np.linalg.norm(edges, axis=1)) < 0.05 * min(work_width, work_height):
                continue
            # Undo OpenCV's pixel-center resize mapping on each axis.
            source_scale = np.array([width / work_width, height / work_height], dtype=np.float32)
            return (corners + 0.5) * source_scale - 0.5

    raise DocumentNotFoundError(
        "No document boundary found. Try a contrasting background with all four corners visible."
    )


def four_point_transform(image: ImageArray, points: ArrayLike) -> ImageArray:
    """Warp the document from the full-resolution BGR source."""
    _validate_image(image)
    ordered = order_points(points)
    top_left, top_right, bottom_right, bottom_left = ordered
    width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    if min(width, height) < 3:
        raise InvalidImageError("The detected document is too small to scan.")

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR)


def render_scan(image: ImageArray, mode: ScanMode) -> ImageArray:
    """Return the requested presentation of a perspective-corrected BGR image."""
    _validate_image(image)
    if mode == "color":
        return image
    if mode not in ("grayscale", "bw"):
        raise ValueError(f"Unknown scan mode: {mode}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "grayscale":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Scale the local threshold window to the scan resolution and keep it odd.
    short_side = min(gray.shape)
    block_size = max(3, min(151, (short_side // 16) | 1))
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 12
    )


def rotate_scan(image: ImageArray, rotation: Rotation = 0) -> ImageArray:
    """Rotate a color or single-channel scan clockwise in exact quarter turns."""
    if rotation == 0:
        return image
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotation not in codes:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees clockwise.")
    return cv2.rotate(image, codes[rotation])


def scan_document(
    image: ImageArray,
    mode: ScanMode = "color",
    *,
    corners: ArrayLike | None = None,
    rotation: Rotation = 0,
) -> ImageArray:
    """Rectify detected or manually selected corners and render the rotated scan."""
    _validate_image(image)
    boundary = detect_document(image) if corners is None else validate_manual_corners(corners, image.shape)
    corrected = four_point_transform(image, boundary)
    return rotate_scan(render_scan(corrected, mode), rotation)
