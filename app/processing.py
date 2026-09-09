"""
processing.py — the "CamScanner magic": find the page, flatten it,
and clean it up.

Pipeline (process_image() is the only function the rest of the app calls):

  1. detect_paper_contour() — grayscale + blur + edge detection, then look
                               for the largest 4-sided contour in the image.
                               Returns 4 corner points, or None if nothing
                               confident was found.
  2. warp_perspective()     — perspective-transform those 4 corners into a
                               flat, straight-on rectangle (like a real scan).
  3. enhance()               — "color" mode: white balance + CLAHE contrast
                               + light sharpening, no grayscale conversion.
                               "bw" mode: adaptive threshold, crisp black
                               text on white, like CamScanner's B&W/document
                               mode — much smaller file size too.

Each function takes/returns a numpy image array (the OpenCV convention)
so they compose cleanly.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

# Teaches Pillow how to open .heic/.heif files (iPhone's default photo
# format) — OpenCV's imread() can't decode them at all.
pillow_heif.register_heif_opener()

PROCESSED_DIR_NAME = "_processed"


def load_image(path: str) -> np.ndarray:
    """
    Loads any supported image (jpg/png/webp/heic/heif) via Pillow rather
    than cv2.imread, for two reasons:
      - Pillow (+ pillow-heif) can decode HEIC/HEIF; OpenCV cannot.
      - ImageOps.exif_transpose() applies the camera's EXIF orientation
        tag, which iPhone photos always carry — without this, photos
        taken in portrait can load sideways or upside down.
    Returns a numpy array in OpenCV's BGR convention.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
            arr = np.array(rgb)
    except Exception as e:
        raise ValueError(f"Could not read image: {path}") from e
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 unordered (x, y) points, return them ordered
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # smallest x+y -> top-left
    rect[2] = pts[np.argmax(s)]  # largest x+y  -> bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # smallest x-y -> top-right
    rect[3] = pts[np.argmax(diff)]  # largest x-y  -> bottom-left
    return rect


def _auto_canny(gray: np.ndarray, low_pct: float = 50, high_pct: float = 93) -> np.ndarray:
    """
    Canny thresholds derived from this image's own gradient-strength
    distribution, rather than fixed constants or overall brightness.
    Brightness-based heuristics get this wrong: a light page on a light
    background has just as weak a gradient as a dark page on a dark
    background, regardless of how bright the image is overall — what
    matters is how strong the page's actual edge is relative to the
    rest of the image's edges (texture, noise, etc).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    nonzero = mag[mag > 1]
    if nonzero.size == 0:
        return cv2.Canny(gray, 50, 150)
    lower = float(np.percentile(nonzero, low_pct))
    upper = float(np.percentile(nonzero, high_pct))
    return cv2.Canny(gray, lower, max(upper, lower + 1))


def _is_plausible_page(quad: np.ndarray, area: float, img_area: float) -> bool:
    """
    Reject quads that technically have 4 points but clearly aren't a page:
    non-convex shapes, slivers, anything with a sharp near-180/near-0
    degree corner (a real page's corners are all close to 90 degrees), or
    the image's own border (a page essentially never fills the frame
    edge-to-edge with zero margin — that's a dense/noisy edge or mask
    map being picked up as a giant rectangle, not an actual page).

    `quad` is a (4, 2) array of corner points (any numeric dtype).
    """
    if area < 0.2 * img_area or area > 0.98 * img_area:
        return False

    contour = quad.reshape(4, 1, 2).astype(np.int32)
    if not cv2.isContourConvex(contour):
        return False

    pts = quad.reshape(4, 2).astype("float64")
    for i in range(4):
        prev_pt, pt, next_pt = pts[i - 1], pts[i], pts[(i + 1) % 4]
        v1, v2 = prev_pt - pt, next_pt - pt
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        if angle < 45 or angle > 135:
            return False
    return True


def _brightness_mask(gray: np.ndarray) -> np.ndarray:
    """
    Binary mask of the brighter of the image's two Otsu-split classes.
    Otsu's "foreground" label is arbitrary — this forces it to always be
    the lighter surface, since a document page is virtually always
    brighter than whatever it's resting on (desk, table, floor, ...).
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mean = cv2.mean(gray, mask=mask)[0]
    bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    return mask if fg_mean >= bg_mean else cv2.bitwise_not(mask)


def _detect_by_brightness(small: np.ndarray, small_area: float):
    """
    Primary page-detection strategy for documents: segment out the
    brightest large region in frame (the white/light page) directly,
    rather than relying on there being a strong enough gradient at its
    boundary — the boundary between a white page and a merely-medium-tone
    background is exactly the case plain edge detection struggles with.
    Returns a (4, 2) float32 quad in `small`'s coordinate space, or None.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = _brightness_mask(gray)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.15 * small_area:
        return None

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype("float32")
        if _is_plausible_page(quad, area, small_area):
            return quad

    # The bright-region mask can have soft/rounded corners (shadow bleed,
    # JPEG noise) that keep approxPolyDP from landing on a clean quad — a
    # minimum-area rotated rectangle around the same region is a solid
    # fallback in that case. But minAreaRect returns a bounding box
    # regardless of the contour's actual shape, so on its own it would
    # launder any oddly-shaped blob (an L-shape, a skewed sliver) into
    # something that looks like a plausible rectangle. Only trust it when
    # the contour is already close to filling its own bounding box — i.e.
    # it was basically a rectangle to begin with.
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")
    box_area = cv2.contourArea(box)
    solidity = area / box_area if box_area > 0 else 0
    if solidity > 0.85 and _is_plausible_page(box, box_area, small_area):
        return box

    return None


def _detect_by_edges(small: np.ndarray, small_area: float):
    """
    Fallback page-detection strategy: gradient/edge-based contour
    detection. Used when the brightness-based approach fails — e.g. a
    background that's just as bright as the page itself, where "biggest
    bright region" doesn't isolate the page at all.
    Returns a (4, 2) float32 quad in `small`'s coordinate space, or None.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Edge-preserving smoothing: knocks down texture/paper-grain noise
    # without blurring the page's actual boundary, unlike a plain Gaussian
    # blur, which softens the edge we're trying to detect.
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = _auto_canny(gray)
    # Closing bridges small gaps in the page outline (e.g. where it's
    # low-contrast against the background for a stretch), so the contour
    # comes out as one closed loop instead of several broken pieces.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        quad = approx.reshape(4, 2).astype("float32")
        if _is_plausible_page(quad, area, small_area):
            return quad

    return None


def detect_paper_contour(img: np.ndarray):
    """
    Look for the page's 4 corners.

    Downscales the image for speed (detection doesn't need full
    resolution), then scales the found corners back up to the original
    image's coordinate space.

    Returns a (4, 2) float32 array of corners in the ORIGINAL image's
    scale, ordered [top-left, top-right, bottom-right, bottom-left],
    or None if no confident page was found.
    """
    h, w = img.shape[:2]
    scale = 800.0 / w if w > 800 else 1.0
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    small_area = small.shape[0] * small.shape[1]

    quad = _detect_by_brightness(small, small_area)
    if quad is None:
        quad = _detect_by_edges(small, small_area)
    if quad is None:
        return None

    return _order_points(quad / scale)


def _pad_corners(corners: np.ndarray, img_shape, pad_ratio: float = 0.025) -> np.ndarray:
    """
    Push each corner outward from the quad's centroid by `pad_ratio` of its
    distance to the centroid, clipped to stay inside the image. The raw
    detected contour tends to hug the page edge tightly (or even inside it
    by a pixel or two), so this keeps a small margin instead of cropping
    right up to — or just past — the page border.
    """
    h, w = img_shape[:2]
    center = corners.mean(axis=0)
    padded = corners + (corners - center) * pad_ratio
    padded[:, 0] = np.clip(padded[:, 0], 0, w - 1)
    padded[:, 1] = np.clip(padded[:, 1], 0, h - 1)
    return padded.astype("float32")


def warp_perspective(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Perspective-warp the region bounded by `corners` (ordered
    [top-left, top-right, bottom-right, bottom-left]) into a flat,
    straight-on rectangle — like the page was scanned rather than
    photographed at an angle.
    """
    (tl, tr, br, bl) = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, matrix, (max_width, max_height))


def _white_balance(img: np.ndarray) -> np.ndarray:
    """Simple gray-world auto white balance."""
    result = img.astype(np.float32)
    avg_b, avg_g, avg_r = (result[:, :, i].mean() for i in range(3))
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    # avoid divide-by-zero on pathological (e.g. solid black) images
    result[:, :, 0] *= avg_gray / max(avg_b, 1e-6)
    result[:, :, 1] *= avg_gray / max(avg_g, 1e-6)
    result[:, :, 2] *= avg_gray / max(avg_r, 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def _sharpen(img: np.ndarray) -> np.ndarray:
    """Light unsharp-mask sharpening."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def _whiten_background(
    img: np.ndarray, kernel_frac: float = 0.35, work_size: int = 500
) -> np.ndarray:
    """
    Flattens uneven lighting/shadow across the page and pushes its
    background toward true white, applied only to the LAB lightness
    channel so the a/b (color) channels pass through untouched and any
    ink, highlighter, or images on the page keep their actual color
    instead of shifting or washing out.

    The background estimate is a morphological closing (a local MAX
    filter), not a blur (local MEAN) — a mean filter's estimate inside a
    large solid-color block is contaminated by the block's own color
    rather than the surrounding white page, which shows up as a visible
    halo/vignette across anything bigger than a few text-stroke widths.
    A max filter instead "reaches past" a dark/colored block to the
    brighter page around it.

    Done on a downscaled copy: morphological closing with the large
    kernel this needs is far too slow at full photo resolution (order of
    a minute on a multi-megapixel image), but the background is
    inherently a smooth, low-frequency thing, so estimating it at low
    resolution and upscaling loses essentially nothing.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    h, w = l.shape

    scale = work_size / max(h, w) if max(h, w) > work_size else 1.0
    small_l = (
        cv2.resize(l, (max(1, int(w * scale)), max(1, int(h * scale))))
        if scale < 1.0
        else l
    )

    k = max(15, int(min(small_l.shape) * kernel_frac) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    small_bg = cv2.morphologyEx(small_l, cv2.MORPH_CLOSE, kernel)
    small_bg = cv2.GaussianBlur(small_bg, (0, 0), sigmaX=k / 4)

    background = (
        cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
        if scale < 1.0
        else small_bg
    )

    l_flat = cv2.divide(l, background, scale=255)
    l_flat = np.clip(l_flat, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_flat, a, b)), cv2.COLOR_LAB2BGR)


def enhance(img: np.ndarray, mode: str = "color") -> np.ndarray:
    """
    mode == "color": white balance (removes color cast) + background
                      whitening (flattens shadows/uneven lighting and
                      pushes the page's white areas to true white,
                      without touching the hue of ink/highlighter/images
                      on the page) + light sharpening.

    mode == "bw":     background-flattened Otsu threshold ANDed with a
                       local adaptive threshold -> crisp black text on a
                       clean white background, like CamScanner's B&W/
                       document mode. Best contrast for plain text pages
                       and the smallest file size.
    """
    if mode == "bw":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Estimate the page's uneven lighting/shadow as a heavily-blurred
        # version of itself, then divide it out. This flattens shadows and
        # gradients across the page BEFORE thresholding, which is what
        # actually gives crisp, even contrast — thresholding raw camera
        # input directly (as before) washes out under any uneven lighting.
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=21)
        normalized = cv2.divide(gray, background, scale=255)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

        # A single global (Otsu) cutoff misses faint marks — light ink
        # stamps, pencil, watermarks — whenever their brightness sits
        # closer to the background than to normal text. ANDing in a local
        # adaptive threshold catches those too: it recomputes the cutoff
        # per neighborhood, so a faint stamp that's merely darker than
        # its own surroundings still gets flagged as ink, without losing
        # Otsu's clean, even result for regular text elsewhere.
        _, otsu = cv2.threshold(
            normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=10,
        )
        thresh = cv2.bitwise_and(otsu, adaptive)

        # Safety margin: perspective-cropping isn't always pixel-perfect,
        # and a sliver of actual background just outside the page can
        # sneak into the crop. That barely shows in "color" mode (it just
        # gets whitened), but here a genuinely dark background pixel
        # thresholds straight to black, showing up as a thin black frame
        # around the page. Force the outermost edge to white rather than
        # rely on the crop being exact.
        h, w = thresh.shape
        by, bx = int(h * 0.012), int(w * 0.012)
        if by:
            thresh[:by, :] = 255
            thresh[-by:, :] = 255
        if bx:
            thresh[:, :bx] = 255
            thresh[:, -bx:] = 255

        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # "color" mode
    img = _white_balance(img)
    img = _whiten_background(img)
    return _sharpen(img)


def process_image(src_path: str, mode: str = "color") -> str:
    """
    Full pipeline for one image: detect the page, flatten it, enhance it.
    Returns the path to the processed output.

    mode: "color" (default) or "bw" — see enhance() above.
    """
    img = load_image(src_path)

    corners = detect_paper_contour(img)
    if corners is not None:
        corners = _pad_corners(corners, img.shape)
        img = warp_perspective(img, corners)

    img = enhance(img, mode=mode)

    # Save processed result next to the original, in a hidden subfolder.
    # Always write as jpg/png regardless of the source format — cv2.imwrite
    # can't encode heic/heif (or webp, on some builds), only decode inputs
    # via load_image() need that range of formats. PNG for "bw" since it's
    # pure black/white and compresses losslessly with no JPEG ringing
    # around text edges; JPEG for "color" photos.
    batch_dir = os.path.dirname(src_path)
    out_dir = os.path.join(batch_dir, PROCESSED_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(src_path))[0]
    if mode == "bw":
        out_path = os.path.join(out_dir, f"{stem}.png")
        cv2.imwrite(out_path, img)
    else:
        out_path = os.path.join(out_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path
