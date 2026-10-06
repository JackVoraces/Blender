"""Check a front render of the head against the reference drawing.

    python tools/compare_reference.py RENDER OUT_PREFIX

RENDER must come from the CAM_ortho_front camera: it frames exactly the region
cropped from the reference here, so the two line up pixel for pixel.  Writes
OUT_PREFIX_side_by_side.png and OUT_PREFIX_overlay.png (a 50/50 blend).
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references" / "head_anatomy_front.webp"
# Crop of the head in the reference, in pixels of the 1116 x 1326 image.  CAM_ortho_front
# (rn_blender/stage.py) frames the same region of head space.
REFERENCE_SIZE = (1116, 1326)
CROP = (300, 100, 945, 960)


def reference_crop():
    image = Image.open(REFERENCE).convert("RGB")
    sx, sy = image.width / REFERENCE_SIZE[0], image.height / REFERENCE_SIZE[1]
    box = (round(CROP[0] * sx), round(CROP[1] * sy), round(CROP[2] * sx), round(CROP[3] * sy))
    return image.crop(box)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    render = Image.open(sys.argv[1]).convert("RGB")
    reference = reference_crop().resize(render.size, Image.LANCZOS)
    prefix = Path(sys.argv[2])
    prefix.parent.mkdir(parents=True, exist_ok=True)

    side = Image.new("RGB", (render.width * 2 + 12, render.height), (24, 24, 28))
    side.paste(reference, (0, 0))
    side.paste(render, (render.width + 12, 0))
    side.save(f"{prefix}_side_by_side.png")
    Image.blend(reference, render, 0.5).save(f"{prefix}_overlay.png")
    print(f"{prefix}_side_by_side.png")
    print(f"{prefix}_overlay.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
