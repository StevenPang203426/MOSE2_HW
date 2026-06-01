"""Patch SAM2Long for MOSEv2 inference.

Usage:
    python patch_sam2long.py SAM2Long

The upstream SAM2Long code can call `.item()` on tensors that contain more than
one value. This happens on MOSE-style videos with multiple objects or multiple
candidate scores and crashes inference with:

    RuntimeError: a Tensor with 2 elements cannot be converted to Scalar
"""

from pathlib import Path
import sys


def patch_sam2_base(sam2long_dir: Path) -> bool:
    target = sam2long_dir / "sam2" / "modeling" / "sam2_base.py"
    if not target.exists():
        raise FileNotFoundError(f"Cannot find {target}")

    text = target.read_text()
    old = """                    if iou.item() > iou_thre and object_score.item() > 0:
                        valid_indices.insert(0, i)
"""
    new = """                    valid_mask = (iou.reshape(-1) > iou_thre) & (object_score.reshape(-1) > 0)
                    if valid_mask.any().item():
                        valid_indices.insert(0, i)
"""

    if new in text:
        return False
    if old not in text:
        raise RuntimeError(
            "Target snippet not found. SAM2Long source may have changed; "
            "patch sam2/modeling/sam2_base.py manually."
        )

    target.write_text(text.replace(old, new))
    return True


def main() -> None:
    sam2long_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    changed = patch_sam2_base(sam2long_dir)
    if changed:
        print(f"Patched {sam2long_dir / 'sam2' / 'modeling' / 'sam2_base.py'}")
    else:
        print("SAM2Long patch already applied")


if __name__ == "__main__":
    main()
