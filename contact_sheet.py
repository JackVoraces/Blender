"""Tile images into a labelled contact sheet.

    python tools/contact_sheet.py OUT.png --cols 5 [--no-labels] IMG [IMG ...]

Labels are the file names without extension.  Run with the project venv.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build(paths, out, cols=5, labels=True, cell=None, pad=6):
    images = [Image.open(p).convert("RGB") for p in paths]
    if cell:
        images = [im.resize((cell, cell), Image.LANCZOS) for im in images]
    w, h = images[0].size
    label_h = 30 if labels else 0
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + label_h + pad) + pad), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    font = _font(20)
    for i, (path, im) in enumerate(zip(paths, images)):
        x = pad + (i % cols) * (w + pad)
        y = pad + (i // cols) * (h + label_h + pad)
        sheet.paste(im, (x, y))
        if labels:
            draw.text((x + 6, y + h + 3), Path(path).stem.replace("_", " "), fill=(235, 235, 235), font=font)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=None, help="resize every image to this square size")
    ap.add_argument("--no-labels", action="store_true")
    args = ap.parse_args()
    print(build(args.images, args.out, args.cols, not args.no_labels, args.cell))


if __name__ == "__main__":
    main()
