"""Generate the WAFAA AI PWA icons as PNG files (no third-party imaging libs).

Draws the lavender gradient tile and the white heart mark with NumPy, then
writes the pixels as PNG using the standard library's zlib/struct modules.

    python3 scripts/generate_icons.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "icons"
SIZES = {"icon-180.png": 180, "icon-192.png": 192, "icon-512.png": 512, "icon-maskable-512.png": 512}


def _png_bytes(rgb: np.ndarray) -> bytes:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def render(size: int, maskable: bool) -> np.ndarray:
    scale = 4  # supersample for smooth edges
    n = size * scale
    ys, xs = np.mgrid[0:n, 0:n]
    u = (xs + 0.5) / n
    v = (ys + 0.5) / n

    # Diagonal lavender gradient (matches the SVG icon).
    stops = [(0.0, (0xC9, 0xB8, 0xFB)), (0.55, (0x9F, 0x80, 0xF2)), (1.0, (0x74, 0x52, 0xD6))]
    t = np.clip((u + v) / 2.0, 0.0, 1.0)
    img = np.zeros((n, n, 3), dtype=np.float64)
    for i in range(len(stops) - 1):
        a_pos, a_col = stops[i]
        b_pos, b_col = stops[i + 1]
        band = (t >= a_pos) & (t <= b_pos)
        local = (t - a_pos) / max(b_pos - a_pos, 1e-9)
        for channel in range(3):
            img[..., channel] = np.where(
                band,
                a_col[channel] + (b_col[channel] - a_col[channel]) * local,
                img[..., channel],
            )
    img = np.where(t >= 1.0, np.array(stops[-1][1], dtype=np.float64), img)

    # Rounded-rectangle alpha (full bleed for maskable icons).
    if maskable:
        inside = np.ones_like(u, dtype=bool)
    else:
        radius = 0.22
        dx = np.maximum(np.abs(u - 0.5) - (0.5 - radius), 0.0)
        dy = np.maximum(np.abs(v - 0.5) - (0.5 - radius), 0.0)
        inside = np.sqrt(dx**2 + dy**2) <= radius

    # White heart: (x^2 + y^2 - 1)^3 - x^2 * y^3 <= 0, scaled and centred.
    cx, cy = 0.5, 0.5
    heart_scale = 0.30 if maskable else 0.34
    hx = (u - cx) / heart_scale
    hy = -(v - cy) / heart_scale
    heart = (hx**2 + hy**2 - 1.0) ** 3 - hx**2 * hy**3 <= 0.0

    alpha = inside.astype(np.float64)
    rgb = img.copy()
    rgb[heart] = rgb[heart] * 0.06 + np.array([255.0, 255.0, 255.0]) * 0.94

    # Composite onto the brand background so the PNG needs no alpha channel.
    bg = np.array([0xF7, 0xF4, 0xFF], dtype=np.float64)
    rgb = rgb * alpha[..., None] + bg * (1.0 - alpha[..., None])

    # Box-downsample back to the requested size.
    rgb = rgb.reshape(size, scale, size, scale, 3).mean(axis=(1, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, size in SIZES.items():
        pixels = render(size, maskable="maskable" in name)
        (OUT_DIR / name).write_bytes(_png_bytes(pixels))
        print(f"wrote {OUT_DIR / name} ({size}x{size})")


if __name__ == "__main__":
    main()
