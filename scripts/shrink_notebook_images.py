"""Store the figures of executed notebooks as 256-colour palette PNGs.

Matplotlib's inline backend writes truecolour RGBA PNGs. In `examples/` those figures were 95 % of
the notebooks' bytes, and a re-execution rewrites every one of them, so each refresh added the whole
set to the history again. Measured 2026-10-24 on the 59 figures of the 23 notebooks: 7.17 MB of
base64 as written, **2.49 MB as palette PNGs** (the files: 7.53 -> 2.85 MB); lossless
re-compression alone saves 2.5 %. An antialiased plot has a few thousand distinct colours, and
median-cut keeps them without visible change: on the largest figure 0.41 % of pixels move by more
than 8/255.

    uv run --no-sync --with pillow python scripts/shrink_notebook_images.py \\
        examples/*.ipynb examples/usecases/*.ipynb

Run it after re-executing a notebook. It is idempotent: a figure already stored as a palette PNG, a
figure with any transparency, and a figure the palette would not make smaller are all left as they
are. A notebook is rewritten only when it round-trips through the JSON layout Jupyter writes
(`indent=1`, sorted keys), so the diff holds image payloads and nothing else.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys

from PIL import Image


def dump(nb: dict) -> str:
    return json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def shrink(payload: str) -> str | None:
    """The palette re-encoding of one base64 PNG, or `None` when it should stay as it is."""
    raw = base64.b64decode(payload)
    image = Image.open(io.BytesIO(raw))
    image.load()
    if image.mode == "P":
        return None
    if image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema()[0] < 255:
        return None
    palette = image.convert("RGB").quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    out = io.BytesIO()
    palette.save(out, "PNG", optimize=True)
    if out.tell() >= len(raw):
        return None
    return base64.b64encode(out.getvalue()).decode("ascii")


def process(path: pathlib.Path, write: bool) -> tuple[int, int, int]:
    """Re-encode one notebook's PNG outputs; returns `(figures changed, bytes before, after)`."""
    text = path.read_text(encoding="utf-8")
    nb = json.loads(text)
    if dump(nb) != text:
        raise SystemExit(f"{path}: not in Jupyter's JSON layout; refusing to rewrite it")
    changed = before = after = 0
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            payload = data.get("image/png")
            if payload is None:
                continue
            as_list = isinstance(payload, list)
            joined = "".join(payload) if as_list else payload
            new = shrink(joined.strip())
            before += len(joined)
            if new is None:
                after += len(joined)
                continue
            new += joined[len(joined.rstrip("\n")) :]
            data["image/png"] = [new] if as_list else new
            after += len(new)
            changed += 1
    if write and changed:
        path.write_text(dump(nb), encoding="utf-8")
    return changed, before, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("notebooks", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--check", action="store_true", help="report what would change, write nothing"
    )
    args = parser.parse_args()
    total_before = total_after = 0
    for path in args.notebooks:
        changed, before, after = process(path, write=not args.check)
        total_before += before
        total_after += after
        if changed:
            print(f"{path}: {changed} figure(s), {before / 1e6:.2f} -> {after / 1e6:.2f} MB")
    verb = "would be" if args.check else "now"
    print(f"figures {total_before / 1e6:.2f} MB, {verb} {total_after / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
