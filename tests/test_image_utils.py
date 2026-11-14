from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytest

from src import image_utils
from src.image_utils import InvalidImageError, decode_image, encode_png


def image_bytes(image, image_format, **options):
    stream = BytesIO()
    image.save(stream, format=image_format, **options)
    return stream.getvalue()


def test_decodes_png_in_bgr_order():
    data = image_bytes(Image.new("RGB", (16, 12), (250, 128, 3)), "PNG")
    result = decode_image(data)
    assert result.shape == (12, 16, 3)
    np.testing.assert_array_equal(result[0, 0], [3, 128, 250])


def test_rotates_jpeg_according_to_exif_orientation():
    image = Image.new("RGB", (16, 24), (255, 0, 0))
    image.paste((0, 0, 255), (0, 12, 16, 24))
    exif = image.getexif()
    exif[274] = 6
    result = decode_image(image_bytes(image, "JPEG", exif=exif))
    assert result.shape == (16, 24, 3)
    assert result[8, 3, 0] > 240
    assert result[8, -4, 2] > 240


def test_composites_transparent_pixels_on_white():
    data = image_bytes(Image.new("RGBA", (10, 10), (0, 0, 0, 0)), "PNG")
    assert np.all(decode_image(data) == 255)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_rejects_unreadable_uploads(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_rejects_an_unsupported_image_format():
    data = image_bytes(Image.new("RGB", (10, 10)), "GIF")
    with pytest.raises(InvalidImageError, match="JPEG, PNG, or WebP"):
        decode_image(data)


def test_rejects_excess_image_dimensions(monkeypatch):
    monkeypatch.setattr(image_utils, "MAX_IMAGE_PIXELS", 99)
    data = image_bytes(Image.new("RGB", (10, 10)), "PNG")
    with pytest.raises(InvalidImageError, match="32 megapixels"):
        decode_image(data)


@pytest.mark.parametrize("shape", [(7, 11), (7, 11, 3)])
def test_png_export_round_trips_without_pixel_changes(shape):
    original = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    png = encode_png(original)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, original)
