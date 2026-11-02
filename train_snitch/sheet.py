"""Comparison sheets without Pillow: ffmpeg decodes/resizes each tile to raw RGB, Python composes the grid and draws
labels with a built-in 5x7 bitmap font, ffmpeg encodes the JPEG. Needs only the `ffmpeg` binary (Homebrew / apt).

  build_sheet(rows, columns, out_path, title=..., tile=256)
    columns: ["BASE", "PREV 500 S0", "CUR 1000 S0", ...]
    rows:    [("#5 GROUP OF SEVEN", [path_or_None, ...]), ...]   (one path per column; None -> grey "N/A" tile)
"""
from __future__ import annotations

import os
import shutil
import subprocess

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

_G = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#", "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.", "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####", "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.####", "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.", "J": "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", "N": "#...#|#...#|##..#|#.#.#|#..##|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.", "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.", "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|#.#.#|.#.#.", "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.", "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####", "3": "#####|...#.|..#..|...#.|....#|#...#|.###.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.", "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.", "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.", "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    " ": ".....|.....|.....|.....|.....|.....|.....", ".": ".....|.....|.....|.....|.....|.##..|.##..",
    ",": ".....|.....|.....|.....|.##..|..#..|.#...", ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "-": ".....|.....|.....|#####|.....|.....|.....", "#": ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.",
    "/": ".....|....#|...#.|..#..|.#...|#....|.....", "(": "...#.|..#..|.#...|.#...|.#...|..#..|...#.",
    ")": ".#...|..#..|...#.|...#.|...#.|..#..|.#...", "+": ".....|..#..|..#..|#####|..#..|..#..|.....",
    "=": ".....|.....|#####|.....|#####|.....|.....", "_": ".....|.....|.....|.....|.....|.....|#####",
    "%": "##...|##..#|...#.|..#..|.#...|#..##|...##", "|": "..#..|..#..|..#..|..#..|..#..|..#..|..#..",
    "'": "..#..|..#..|.#...|.....|.....|.....|.....", "?": ".###.|#...#|....#|...#.|..#..|.....|..#..",
    "&": ".##..|#..#.|#.#..|.#...|#.#.#|#..#.|.##.#", "@": ".###.|#...#|#.###|#.#.#|#.###|#....|.###.",
}
GLYPH = {k: [r for r in v.split("|")] for k, v in _G.items()}
SCALE = 2
ADV = 6 * SCALE      # advance per character (5 px glyph + 1 px gap)
LINE_H = 9 * SCALE   # 7 px glyph + padding
BG = (255, 255, 255)
FG = (20, 20, 20)
NA = (170, 170, 170)


class Canvas:
    def __init__(self, w: int, h: int, color=BG):
        self.w, self.h = w, h
        self.buf = bytearray(bytes(color) * (w * h))

    def rect(self, x, y, w, h, color):
        row = bytes(color) * w
        for yy in range(max(0, y), min(self.h, y + h)):
            o = (yy * self.w + x) * 3
            self.buf[o:o + w * 3] = row

    def paste_rgb(self, x, y, w, h, data: bytes):
        for r in range(h):
            o = ((y + r) * self.w + x) * 3
            self.buf[o:o + w * 3] = data[r * w * 3:(r + 1) * w * 3]

    def text(self, x, y, s: str, color=FG, max_w: int | None = None):
        s = str(s).upper()
        if max_w is not None and len(s) * ADV > max_w:
            s = s[:max(1, max_w // ADV - 2)] + ".."
        px = bytes(color) * SCALE
        for i, ch in enumerate(s):
            g = GLYPH.get(ch, GLYPH["?"])
            gx = x + i * ADV
            if gx + 5 * SCALE > self.w:
                break
            for r, line in enumerate(g):
                for c, bit in enumerate(line):
                    if bit != "#":
                        continue
                    for dy in range(SCALE):
                        yy = y + r * SCALE + dy
                        if 0 <= yy < self.h:
                            o = (yy * self.w + gx + c * SCALE) * 3
                            self.buf[o:o + 3 * SCALE] = px


def text_width(s: str) -> int:
    return len(str(s)) * ADV


def decode_tile(path: str, tile: int, timeout: float = 60) -> bytes | None:
    """Image -> tile x tile RGB bytes (aspect kept, grey padding). None when ffmpeg cannot read it."""
    vf = (f"scale={tile}:{tile}:force_original_aspect_ratio=decrease,"
          f"pad={tile}:{tile}:(ow-iw)/2:(oh-ih)/2:color=0xb0b0b0,format=rgb24")
    try:
        r = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", path, "-frames:v", "1", "-vf", vf,
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode or len(r.stdout) != tile * tile * 3:
        return None
    return r.stdout


def encode_jpeg(cv: Canvas, out_path: str, quality: int = 3, timeout: float = 120) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp.jpg"
    r = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{cv.w}x{cv.h}", "-i", "-", "-frames:v", "1", "-q:v", str(quality), tmp],
                       input=bytes(cv.buf), capture_output=True, timeout=timeout)
    if r.returncode or not os.path.exists(tmp):
        raise RuntimeError(f"ffmpeg encode failed: {r.stderr.decode(errors='replace')[-300:]}")
    os.replace(tmp, out_path)


def build_sheet(rows: list[tuple[str, list[str | None]]], columns: list[str], out_path: str, title: str = "",
                tile: int = 256, gap: int = 6, quality: int = 3) -> dict:
    """Grid sheet: title bar, column labels, then per row a label strip + one tile per column."""
    ncol = len(columns)
    pad = 8
    w = pad * 2 + ncol * tile + (ncol - 1) * gap
    head = LINE_H * 2 + 8
    row_h = LINE_H + tile + gap
    h = head + len(rows) * row_h + pad
    cv = Canvas(w, h)
    cv.rect(0, 0, w, LINE_H + 4, (235, 235, 235))
    cv.text(pad, 4, title, max_w=w - 2 * pad)
    for c, lab in enumerate(columns):
        cv.text(pad + c * (tile + gap), LINE_H + 8, lab, max_w=tile)
    missing = 0
    for r, (rlab, paths) in enumerate(rows):
        y = head + r * row_h
        cv.rect(0, y, w, LINE_H, (245, 240, 225))
        cv.text(pad, y + 2, rlab, max_w=w - 2 * pad)
        for c in range(ncol):
            p = paths[c] if c < len(paths) else None
            x, ty = pad + c * (tile + gap), y + LINE_H
            data = decode_tile(p, tile) if p else None
            if data is None:
                missing += 1
                cv.rect(x, ty, tile, tile, NA)
                cv.text(x + tile // 2 - text_width("N/A") // 2, ty + tile // 2 - 7, "N/A", color=(90, 90, 90))
            else:
                cv.paste_rgb(x, ty, tile, tile, data)
    encode_jpeg(cv, out_path, quality=quality)
    return {"path": out_path, "w": w, "h": h, "tiles": ncol * len(rows), "missing": missing}
