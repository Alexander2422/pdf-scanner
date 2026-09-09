"""
pdf_export.py — turns a list of processed image paths into PDF file(s).

Uses img2pdf rather than e.g. reportlab or Pillow's PDF saver because:
  - it's lossless (doesn't re-encode/recompress your JPEGs)
  - it correctly embeds images at their real DPI so pages come out the
    right physical size instead of tiny/huge
  - it's a tiny, fast, single-purpose library — good fit for a Pi
"""

import os
import img2pdf


def images_to_pdf(image_paths: list[str], out_dir: str, mode: str = "single") -> list[str]:
    """
    image_paths : processed images, in the order they should appear
    out_dir     : where to write the resulting PDF(s)
    mode        : "single"   -> one combined multi-page PDF
                  "separate" -> one PDF per image

    Returns a list of output PDF filenames (not full paths) for the
    Flask route to build download links from.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []

    if mode == "separate":
        for path in image_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            out_name = f"{name}.pdf"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, "wb") as f:
                f.write(img2pdf.convert(path))
            results.append(out_name)
    else:
        out_name = "scan.pdf"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(img2pdf.convert(image_paths))
        results.append(out_name)

    return results
