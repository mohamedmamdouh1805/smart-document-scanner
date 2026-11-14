"""Validate uploads and convert between Pillow and OpenCV image formats."""

from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

ImageArray = NDArray[np.uint8]
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")


class InvalidImageError(ValueError):
    """The supplied image cannot be safely processed."""


def decode_image(data: bytes) -> ImageArray:
    """Decode JPEG, PNG or WebP bytes into an EXIF-oriented BGR image."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(f"Choose a nonempty image up to {MAX_UPLOAD_MB} MB.")

    try:
        with Image.open(BytesIO(data), formats=SUPPORTED_FORMATS) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Choose an image with at most 32 megapixels.")
            if min(image.size) < 3:
                raise InvalidImageError("The image must be at least 3 × 3 pixels.")

            oriented = ImageOps.exif_transpose(image)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                oriented = Image.alpha_composite(background, rgba)
            rgb = np.asarray(oriented.convert("RGB"))
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("Choose an image with at most 32 megapixels.") from exc
    except UnidentifiedImageError as exc:
        raise InvalidImageError("Use a readable JPEG, PNG, or WebP image.") from exc
    except (OSError, SyntaxError) as exc:
        raise InvalidImageError("The image could not be read. Check that it opens normally.") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_png(image: ImageArray) -> bytes:
    """Encode a BGR or grayscale output as a lossless PNG."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise InvalidImageError("The scan could not be encoded as a PNG.")
    return encoded.tobytes()
