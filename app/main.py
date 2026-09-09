"""
main.py — Flask app entrypoint.

Responsibilities of this file ONLY:
  - Web routes (upload page, upload handler, export handler, download)
  - Session/batch bookkeeping (which images belong to which upload batch)

It deliberately does NOT contain any image-processing or PDF logic —
that lives in processing.py and pdf_export.py. Keeping routes "dumb"
like this makes it much easier to test/replace the processing pipeline
later without touching the web layer.
"""

import os
import uuid
import cv2
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, session, flash, Response
)

from processing import process_image, load_image
from pdf_export import images_to_pdf

# Browsers other than Safari can't render HEIC/HEIF in an <img> tag at all,
# so thumbnails for these get converted to JPEG on the fly (see
# uploaded_file() below) rather than served as the raw upload.
HEIC_EXTENSIONS = {"heic", "heif"}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SCAN_APP_SECRET", "dev-secret-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic", "heif"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def list_batch_images(batch_dir: str) -> list[str]:
    """
    Uploaded image files directly in a batch dir, excluding subfolders
    like processing.py's "_processed" cache from a previous export.
    """
    if not os.path.isdir(batch_dir):
        return []
    return sorted(
        name for name in os.listdir(batch_dir)
        if allowed_file(name) and os.path.isfile(os.path.join(batch_dir, name))
    )


def get_batch_id() -> str:
    """
    Each 'session' of uploads (before you hit Export) is a batch.
    We use a folder-per-batch so multiple people on the LAN uploading
    at the same time don't collide with each other's files.
    """
    if "batch_id" not in session:
        session["batch_id"] = uuid.uuid4().hex[:12]
    batch_dir = os.path.join(UPLOAD_DIR, session["batch_id"])
    os.makedirs(batch_dir, exist_ok=True)
    return session["batch_id"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Show the upload page + thumbnails of whatever is in the current batch."""
    batch_id = get_batch_id()
    batch_dir = os.path.join(UPLOAD_DIR, batch_id)
    images = list_batch_images(batch_dir)
    return render_template("index.html", images=images, batch_id=batch_id)


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts one or more files from the <input type="file" multiple> field.
    Files are saved as-is (no processing yet) into the current batch folder.
    Processing happens later, at export time, not at upload time — this way
    if we change enhancement settings, we haven't destroyed the originals.
    """
    batch_id = get_batch_id()
    batch_dir = os.path.join(UPLOAD_DIR, batch_id)

    files = request.files.getlist("images")
    if not files:
        flash("No files selected.")
        return redirect(url_for("index"))

    for f in files:
        if f and f.filename and allowed_file(f.filename):
            # Prefix with a short uuid to avoid collisions/overwrites
            safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
            f.save(os.path.join(batch_dir, safe_name))

    return redirect(url_for("index"))


@app.route("/delete/<batch_id>/<filename>", methods=["POST"])
def delete_image(batch_id, filename):
    """Remove a single image from the batch before export."""
    path = os.path.join(UPLOAD_DIR, batch_id, filename)
    if os.path.isfile(path):
        os.remove(path)
    return redirect(url_for("index"))


@app.route("/export", methods=["POST"])
def export():
    """
    Runs every image in the batch through the processing pipeline,
    then bundles the result into PDF(s) according to the user's choice:
      - "single"   -> one multi-page PDF
      - "separate" -> one PDF per image
    Returns links to whatever PDF(s) got created.
    """
    batch_id = get_batch_id()
    batch_dir = os.path.join(UPLOAD_DIR, batch_id)
    mode = request.form.get("export_mode", "single")  # "single" | "separate"
    scan_mode = request.form.get("scan_mode", "color")  # "color" | "bw"
    order_raw = request.form.get("order", "")

    filenames = list_batch_images(batch_dir)
    if not filenames:
        flash("Nothing to export yet — upload some images first.")
        return redirect(url_for("index"))

    # The gallery's drag-to-reorder only matters for a single combined
    # PDF — each "separate" PDF is independent regardless of page order,
    # so that mode keeps the plain filename-sorted listing.
    if mode == "single" and order_raw:
        requested_order = [name for name in order_raw.split(",") if name]
        ordered = [name for name in requested_order if name in filenames]
        remaining = [name for name in filenames if name not in ordered]
        filenames = ordered + remaining

    # 1. Run each image through the processing pipeline: detect the page,
    #    flatten its perspective, and enhance contrast/color.
    processed_paths = []
    for name in filenames:
        src_path = os.path.join(batch_dir, name)
        processed_path = process_image(src_path, mode=scan_mode)
        processed_paths.append(processed_path)

    # 2. Bundle into PDF(s)
    out_batch_dir = os.path.join(OUTPUT_DIR, batch_id)
    os.makedirs(out_batch_dir, exist_ok=True)

    pdf_files = images_to_pdf(processed_paths, out_batch_dir, mode=mode)

    return render_template("export_done.html", batch_id=batch_id, pdf_files=pdf_files)


@app.route("/download/<batch_id>/<filename>")
def download(batch_id, filename):
    out_batch_dir = os.path.join(OUTPUT_DIR, batch_id)
    return send_from_directory(out_batch_dir, filename, as_attachment=True)


@app.route("/uploads/<batch_id>/<filename>")
def uploaded_file(batch_id, filename):
    """
    Serves an uploaded image so the HTML page can show it as a thumbnail.

    HEIC/HEIF get converted to JPEG on the fly here — only Safari can
    render those formats directly in an <img> tag, so serving them as-is
    would leave a broken image icon in every other browser. The original
    upload on disk is untouched; this only affects what's sent for display.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in HEIC_EXTENSIONS:
        path = os.path.join(UPLOAD_DIR, batch_id, filename)
        img = load_image(path)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), mimetype="image/jpeg")
    return send_from_directory(os.path.join(UPLOAD_DIR, batch_id), filename)


if __name__ == "__main__":
    # 0.0.0.0 so it's reachable from other devices on the LAN, not just localhost.
    # debug=True exposes the interactive Werkzeug debugger (arbitrary code
    # execution) to anyone who can reach the port, so it's off by default —
    # set SCAN_APP_DEBUG=1 for local development only.
    debug = os.environ.get("SCAN_APP_DEBUG") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
