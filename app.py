"""Streamlit interface for the Smart Document Scanner."""

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.image_utils import MAX_UPLOAD_MB, ImageArray, InvalidImageError, decode_image, encode_png
from src.scanner import DocumentNotFoundError, Rotation, ScanMode, detect_document, scan_document

MODES: dict[str, ScanMode] = {
    "Color": "color",
    "Enhanced grayscale": "grayscale",
    "Black and white": "bw",
}
ROTATIONS: dict[str, Rotation] = {
    "0°": 0,
    "90° clockwise": 90,
    "180°": 180,
    "270° clockwise": 270,
}


def manual_corner_editor(image: ImageArray, image_key: str) -> list[tuple[int, int]]:
    """Present pixel inputs and render a numbered overlay on a display copy."""
    height, width = image.shape[:2]
    try:
        detected = detect_document(image)
        initial = np.clip(np.rint(detected), [0, 0], [width - 1, height - 1]).astype(int)
        st.caption("Starting points are from automatic detection.")
    except DocumentNotFoundError:
        left, top = width // 8, height // 8
        right, bottom = width - 1 - left, height - 1 - top
        initial = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
        st.caption("Automatic detection found no boundary. Move the four starting points onto the page.")

    st.caption("Match the numbered markers to the page corners. X moves right and Y moves down, in source pixels.")
    points = []
    for index, (start_x, start_y) in enumerate(initial, start=1):
        x_column, y_column = st.columns(2)
        x = x_column.number_input(
            f"Corner {index} X", 0, width - 1, int(start_x), key=f"{image_key}-{index}-x"
        )
        y = y_column.number_input(
            f"Corner {index} Y", 0, height - 1, int(start_y), key=f"{image_key}-{index}-y"
        )
        points.append((x, y))
    return points


def corner_preview(image: ImageArray, points: list[tuple[int, int]]) -> ImageArray:
    """Scale the overlay to the preview while keeping the selected source pixels intact."""
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / max(width, height))
    preview_width, preview_height = max(1, round(width * scale)), max(1, round(height * scale))
    if scale < 1:
        preview = cv2.resize(image, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    else:
        preview = image.copy()

    # Project source pixel centers to preview pixel centers, then clamp to the display.
    ratio = np.array([preview_width / width, preview_height / height])
    scaled = np.rint((np.array(points) + 0.5) * ratio - 0.5)
    markers = np.clip(scaled, [0, 0], [preview_width - 1, preview_height - 1]).astype(np.int32)
    color = (255, 230, 30)
    hull = cv2.convexHull(markers)
    cv2.polylines(preview, [hull], True, color, 2, cv2.LINE_AA)
    for index, (x, y) in enumerate(markers, start=1):
        cv2.circle(preview, (x, y), 6, color, -1, cv2.LINE_AA)
        origin = (int(np.clip(x + 9, 0, max(0, preview_width - 14))),
                  int(np.clip(y - 8, min(15, preview_height - 1), preview_height - 1)))
        cv2.putText(preview, str(index), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, str(index), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return preview


def main() -> None:
    st.set_page_config(page_title="Smart Document Scanner", layout="wide")
    st.title("Smart Document Scanner")
    st.write("Straighten a photographed document and download it as a PNG.")

    controls, previews = st.columns([1, 3], gap="large")
    with controls:
        uploaded = st.file_uploader(
            "Document photo",
            type=["jpg", "jpeg", "png", "webp"],
        )
        st.caption(f"Up to {MAX_UPLOAD_MB} MB and 32 megapixels. Include all four corners.")
        selected = st.radio("Scan style", list(MODES))
        rotation = st.selectbox("Output rotation", list(ROTATIONS))
        boundary_mode = st.radio("Document boundary", ["Automatic", "Manual corners"])

    if uploaded is None:
        return

    with previews:
        try:
            data = uploaded.getvalue()
            original = decode_image(data)
        except InvalidImageError as exc:
            st.error(str(exc))
            return
        except (cv2.error, MemoryError):
            st.error("The photo could not be processed. Try a smaller image.")
            return

        before, after = st.columns(2, gap="medium")
        corners = None
        with before:
            st.subheader("Original")
            if boundary_mode == "Manual corners":
                preview_slot = st.empty()
                try:
                    corners = manual_corner_editor(original, sha256(data).hexdigest())
                    preview_slot.image(corner_preview(original, corners), channels="BGR", width="stretch")
                except (cv2.error, MemoryError):
                    st.error("The corner editor could not load this photo. Try a smaller image.")
                    return
            else:
                st.image(original, channels="BGR", width="stretch")

        with after:
            st.subheader("Scan")
            try:
                with st.spinner("Preparing the scan…"):
                    result = scan_document(
                        original, MODES[selected], corners=corners, rotation=ROTATIONS[rotation]
                    )
                    download = encode_png(result)
            except DocumentNotFoundError as exc:
                st.warning(str(exc))
                st.info("Choose Manual corners to outline the document yourself.")
                return
            except InvalidImageError as exc:
                st.error(str(exc))
                return
            except (cv2.error, MemoryError):
                st.error("The scan could not be generated. Try a smaller or clearer photo.")
                return

            height, width = result.shape[:2]
            st.image(result, channels="BGR" if result.ndim == 3 else "RGB", width="stretch")
            status = "Document detected" if corners is None else "Manual boundary applied"
            st.caption(f"{status} · {width} × {height} px")
            st.download_button(
                "Download PNG",
                data=download,
                file_name=f"{Path(uploaded.name).stem}-scan.png",
                mime="image/png",
                type="primary",
                on_click="ignore",
            )


if __name__ == "__main__":
    main()
