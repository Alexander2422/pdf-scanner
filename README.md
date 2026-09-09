# Scan App — backbone

A self-hosted "CamScanner"-style app: upload photos of documents from any
device on your LAN, get back clean PDFs.

## Current state

Everything runs end to end: upload, batch management, export, download —
and `app/processing.py` now does the real "CamScanner" work:

1. **Page detection** — edge detection + contour finding locates the
   page's 4 corners in the photo (falls back to using the full frame
   untouched if no confident quadrilateral is found).
2. **Perspective warp** — flattens those 4 corners into a straight,
   top-down rectangle, like a real scan instead of an angled photo.
3. **Enhancement** — two selectable styles, picked per export in the UI:
   - **Color**: gray-world white balance + CLAHE contrast boost (applied
     to luminance only, so colors don't wash out) + light sharpening.
   - **Black & white**: adaptive threshold for crisp black text on a
     clean white background — CamScanner's "document" mode equivalent,
     and much smaller PDFs.

## Project structure

```
scan-app/
├── Dockerfile              # ARM-compatible (works on Raspberry Pi)
├── docker-compose.yml      # run with `docker compose up`
├── requirements.txt
├── README.md
└── app/
    ├── main.py             # Flask routes only — upload, export, download
    ├── processing.py       # STUB: perspective correction + enhancement
    ├── pdf_export.py       # Bundles processed images into PDF(s)
    ├── templates/
    │   ├── index.html      # Upload page + thumbnail gallery
    │   └── export_done.html
    ├── static/style.css
    ├── uploads/            # (gitignore this) raw uploaded images, per batch
    └── output/             # (gitignore this) generated PDFs, per batch
```

## How the pieces fit together

1. **`main.py`** — the only file that knows about HTTP. It saves uploads
   into a per-batch folder (`uploads/<batch_id>/`), and on export calls
   `processing.process_image()` on each file, then
   `pdf_export.images_to_pdf()` on the results.
2. **`processing.py`** — the only file that will know about OpenCV. Its
   pipeline (already shaped, not yet implemented):
   `detect_paper_contour → warp_perspective → enhance`.
3. **`pdf_export.py`** — the only file that knows about PDF creation, using
   `img2pdf` for lossless, correctly-scaled output.

Splitting it this way means we can rewrite the OpenCV internals later
without touching Flask routes or PDF logic at all.

## Running it locally (before Docker)

```bash
cd scan-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd app
python main.py
```

Visit `http://localhost:5000`.

## Running with Docker (this is how it'll run on the Pi)

```bash
cd scan-app
docker compose up --build
```

Then from any device on your LAN, visit `http://<pi-ip-address>:5000`.

Uploaded images and generated PDFs are persisted to `./data/uploads` and
`./data/output` on the host (via the volume mounts in
`docker-compose.yml`), so they survive container restarts.

## What's NOT built yet (next steps)

- [ ] Drag-and-drop upload UX (currently a plain file input)
- [ ] Optional: manual corner-adjustment UI (you chose automatic-only for
      now, so this isn't required, but the code is structured so it could
      be added later without much rework)
- [ ] Basic auth if you want to restrict who on your LAN can use it

## Notes for the Pi

- `opencv-python-headless` has prebuilt wheels for ARM, so `pip install`
  should work directly on the Pi without needing to compile from source —
  but first Docker builds can still take a few minutes on a Pi's CPU.
- If you're on a Pi Zero/older Pi with very limited RAM, building the
  Docker image directly on the Pi may be slow; consider building it on
  your dev machine with `docker buildx build --platform linux/arm64` and
  pushing to a registry (or just `docker save`/`docker load` it over).
