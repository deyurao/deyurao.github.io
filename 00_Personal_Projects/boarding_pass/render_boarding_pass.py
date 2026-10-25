#!/usr/bin/env python3
"""CX 327 boarding pass for a 3.68\" Spectra 6 panel (528×792).

The QR code is not re-encoded. Modules are the black/white cells sampled
from the issued pass, then scaled by an integer factor with nearest-neighbor.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 528, 792
QR_MODULES = 49
QR_SCALE = 8  # each original cell becomes an 8×8 block; 0.75 mm on this panel
QUIET_MODULES = 4

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PALETTE = (BLACK, WHITE, YELLOW, RED, BLUE, GREEN)

# Approximate Spectra 6 pigments, for on-screen preview only.
PIGMENT = {
    BLACK: (0, 0, 0),
    WHITE: (186, 188, 189),
    YELLOW: (208, 190, 71),
    RED: (156, 72, 75),
    BLUE: (61, 59, 94),
    GREEN: (58, 91, 70),
}

BOLD = "/usr/share/fonts/truetype/macos/Inter-Bold.ttf"
SEMI = "/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf"
MED = "/usr/share/fonts/truetype/macos/Inter-Medium.ttf"

# 49×49 modules, row-major, packed MSB first. 1 = black.
# Sampled from the issued boarding-pass image; OpenCV and a center-window
# sample agreed on every cell, and the matrix decodes to the original payload.
MODULE_HEX = (
    "fe708eaa30bfc136a8b9bbd06eafbb28ca6bb74caaeae0a5dba7f77eaec2ec13"
    "eed11af107faaaaaaaaafe01299c799a00efc223e8a962584fb86aeeaae1d133"
    "337723b4b9ee19feab2f7d5deaeba2aa27aaae244aab7f8efb36aa2eb38aa8b1"
    "92bb5eda27eaed28926bb82acaa9fac8ab3329633f042399dc2ba82f2ac8ade2"
    "8d6404568c8aa2bee89fbaabef51499c7d9b182aea3eaec8a89c6bb11a4ac5e3"
    "f334fb393f2a9aeea5d3be21ff5dceecca2c0e6aa3a442a51c8efcb2baab12aa"
    "ae79d8e8473a2b28aea12cffb3aaa2a8cec2a89b230eb71622c9990c2338aaf6"
    "acaa29464468a88aa91d8898f38aaee3d9957b90e8e3e22feaa8f8804bb11a42"
    "c4ffa330aa292bb05eec461e712badddffab8fedd1aaa7aa4e2eeb6ee3726ae9"
    "052aa3389a8afea2332bea8180"
)
PAYLOAD = (
    "M1RAO/DEYU            ECTNGH3 SHAHKGCX 0327 298Y044D0014 165>6320WW6296BCX"
    "                                        2A0272160918863 1CX AS 326224920"
    "        20KN3AS"
)

SUPERSAMPLE = 4
INK_THRESHOLD = 128


def modules_from_hex() -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(MODULE_HEX), dtype=np.uint8)
    bits = np.unpackbits(raw)[: QR_MODULES * QR_MODULES]
    return bits.reshape(QR_MODULES, QR_MODULES)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def measure(text: str, path: str, size: int, tracking: float = 0) -> tuple[float, float]:
    f = font(path, size * SUPERSAMPLE)
    if not text:
        return 0.0, 0.0
    if tracking == 0 or len(text) == 1:
        box = f.getbbox(text)
        return (box[2] - box[0]) / SUPERSAMPLE, (box[3] - box[1]) / SUPERSAMPLE
    width = 0.0
    height = 0.0
    for i, ch in enumerate(text):
        box = f.getbbox(ch)
        width += (box[2] - box[0]) / SUPERSAMPLE
        height = max(height, (box[3] - box[1]) / SUPERSAMPLE)
        if i != len(text) - 1:
            width += tracking
    return width, height


class Board:
    def __init__(self) -> None:
        self.arr = np.full((H, W, 3), 255, dtype=np.uint8)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            return
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 < x1 and y0 < y1:
            self.arr[y0:y1, x0:x1] = color

    def text(
        self,
        x: float,
        y: float,
        text: str,
        path: str,
        size: int,
        color: tuple[int, int, int],
        anchor: str = "lt",
        tracking: float = 0,
    ) -> tuple[int, int, int, int]:
        """Draw solid (not antialiased) text. anchor is two chars: l/c/r and t/m/b."""
        if not text:
            return (int(x), int(y), 0, 0)
        f = font(path, size * SUPERSAMPLE)
        gap = int(round(tracking * SUPERSAMPLE))
        glyphs = []
        width = 0
        top = None
        bottom = None
        for i, ch in enumerate(text):
            box = f.getbbox(ch)
            # Whitespace can report a zero-height box on the baseline. Keep it
            # for advance, but don't let it pull the line's ink box.
            if box[3] > box[1]:
                top = box[1] if top is None else min(top, box[1])
                bottom = box[3] if bottom is None else max(bottom, box[3])
            glyphs.append((ch, box))
            width += max(0, box[2] - box[0])
            if i != len(text) - 1:
                width += gap
        if top is None:
            return (int(x), int(y), 0, 0)
        height = bottom - top
        pad = SUPERSAMPLE * 2
        mask = Image.new("L", (width + pad * 2, height + pad * 2), 0)
        draw = ImageDraw.Draw(mask)
        pen = pad
        for i, (ch, box) in enumerate(glyphs):
            draw.text((pen - box[0], pad - top), ch, font=f, fill=255)
            pen += box[2] - box[0]
            if i != len(text) - 1:
                pen += gap
        big = np.asarray(mask)
        bh, bw = big.shape
        bh = bh - (bh % SUPERSAMPLE)
        bw = bw - (bw % SUPERSAMPLE)
        small = big[:bh, :bw].reshape(bh // SUPERSAMPLE, SUPERSAMPLE, bw // SUPERSAMPLE, SUPERSAMPLE)
        ink = small.mean(axis=(1, 3)) >= INK_THRESHOLD
        tw, th = width / SUPERSAMPLE, height / SUPERSAMPLE
        ax, ay = anchor[0], anchor[1]
        if ax == "c":
            left = x - tw / 2
        elif ax == "r":
            left = x - tw
        else:
            left = x
        if ay == "m":
            top_px = y - th / 2
        elif ay == "b":
            top_px = y - th
        else:
            top_px = y
        # mask origin is `pad` supersampled pixels before the glyph box
        x0 = int(round(left - pad / SUPERSAMPLE))
        y0 = int(round(top_px - pad / SUPERSAMPLE))
        self._stamp(ink, x0, y0, color)
        return (int(round(left)), int(round(top_px)), int(round(tw)), int(round(th)))

    def _stamp(self, ink: np.ndarray, x0: int, y0: int, color: tuple[int, int, int]) -> None:
        ih, iw = ink.shape
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = min(iw, W - x0)
        sy1 = min(ih, H - y0)
        if sx0 >= sx1 or sy0 >= sy1:
            return
        patch = ink[sy0:sy1, sx0:sx1]
        ys, xs = np.nonzero(patch)
        self.arr[y0 + sy0 + ys, x0 + sx0 + xs] = color

    def arrow(self, x: int, y: int, length: int, head: int, color: tuple[int, int, int]) -> None:
        """Right-pointing arrow. (x, y) is the left-center."""
        head = max(5, head)
        if head % 2 == 0:
            head += 1
        shaft_w = max(1, length - (head + 1) // 2)
        shaft_h = max(2, head // 5)
        if shaft_h % 2 == 0:
            shaft_h += 1
        self.rect(x, y - shaft_h // 2, shaft_w, shaft_h, color)
        for dx in range(head):
            hh = head - 2 * dx
            if hh <= 0:
                break
            self.rect(x + shaft_w + dx, y - hh // 2, 1, hh, color)

    def image(self) -> Image.Image:
        return Image.fromarray(self.arr, "RGB")


def build(mods: np.ndarray) -> Image.Image:
    b = Board()
    qr_px = QR_MODULES * QR_SCALE
    quiet = QUIET_MODULES * QR_SCALE
    qr_x = (W - qr_px) // 2
    qr_y = H - quiet - qr_px
    content_bottom = qr_y - quiet  # exclusive; below this must stay white

    # Brand stripe, then a black bar. White text on black is the readable pair;
    # the green stripe is a solid pigment field (text-on-green is too dim).
    b.rect(0, 0, W, 10, GREEN)
    b.rect(0, 10, W, 42, BLACK)

    econ = "ECONOMY"
    ew, eh = measure(econ, BOLD, 13)
    chip_w, chip_h = int(ew) + 16, int(eh) + 8
    chip_x = W - 12 - chip_w
    chip_y = 10 + (42 - chip_h) // 2
    b.rect(chip_x, chip_y, chip_w, chip_h, YELLOW)
    b.text(chip_x + chip_w / 2, chip_y + chip_h / 2, econ, BOLD, 13, BLACK, "cm")
    b.text(16, 31, "CATHAY PACIFIC", SEMI, 18, WHITE, "lm", tracking=1.4)

    # Route + seat badge.
    y = 64
    seat_size = 46
    seat_w, seat_h = measure("44D", BOLD, seat_size)
    label_w, label_h = measure("SEAT", SEMI, 12)
    badge_w = int(max(seat_w, label_w) + 24)
    stack_h = 90
    badge_x = W - 16 - badge_w
    badge_y = y
    b.rect(badge_x, badge_y, badge_w, stack_h, YELLOW)
    _, seat_label_top, _, seat_label_h = b.text(
        badge_x + badge_w / 2, badge_y + 8, "SEAT", SEMI, 12, BLACK, "ct"
    )
    seat_mid = (seat_label_top + seat_label_h + 6 + badge_y + stack_h - 8) / 2
    b.text(badge_x + badge_w / 2, seat_mid, "44D", BOLD, seat_size, BLACK, "cm")

    code_size = 50
    sha_w, sha_h = measure("SHA", BOLD, code_size)
    hkg_w, hkg_h = measure("HKG", BOLD, code_size)
    arrow_len, arrow_head = 36, 17
    gap = 12
    route_w = sha_w + gap + arrow_len + gap + hkg_w
    route_left = 16
    route_right_limit = badge_x - 10
    if route_left + route_w > route_right_limit:
        raise SystemExit(f"route too wide: {route_w:.0f} > {route_right_limit - route_left}")
    code_y = y + 2
    b.text(route_left, code_y, "SHA", BOLD, code_size, BLACK, "lt")
    arrow_x = int(round(route_left + sha_w + gap))
    arrow_y = int(round(code_y + sha_h / 2))
    b.arrow(arrow_x, arrow_y, arrow_len, arrow_head, BLUE)
    b.text(route_left + sha_w + gap + arrow_len + gap, code_y, "HKG", BOLD, code_size, BLACK, "lt")

    b.text(16, y + 58, "CX 327", BOLD, 18, BLACK, "lt")
    b.text(16, y + 80, "Shanghai to Hong Kong", MED, 15, BLACK, "lt")

    y = badge_y + stack_h + 8  # 168
    b.rect(16, y, W - 32, 2, BLACK)
    y += 10

    # Four primary fields.
    columns = [
        ("DATE", "25 OCT", "SUN 2026"),
        ("DEPARTS", "22:30", "10:30 PM"),
        ("BOARDING", "22:00", "GROUP 6"),
        ("PASS GATE", None, "TERMINAL 1"),
    ]
    col_left = 16
    col_right = W - 16
    col_w = (col_right - col_left) / 4
    value_size = 28
    _, value_h = measure("22:30", BOLD, value_size)
    for i, (label, value, sub) in enumerate(columns):
        cx = col_left + i * col_w
        b.text(cx, y, label, SEMI, 12, BLACK, "lt")
        value_top = y + 16
        if value is None:
            # Unassigned gate on the issued pass. Red bar flags it.
            bar_w, bar_h = 26, 5
            bar_y = int(value_top + value_h / 2 - bar_h / 2)
            b.rect(int(cx), bar_y, 8, bar_h, RED)
            b.rect(int(cx) + 11, bar_y, bar_w, bar_h, BLACK)
        else:
            b.text(cx, value_top, value, BOLD, value_size, BLACK, "lt")
        b.text(cx, value_top + value_h + 5, sub, MED, 13, BLACK, "lt")
    y = int(y + 16 + value_h + 5 + 14 + 8)

    _, _, _, name_h = b.text(16, y, "RAO / DEYU", BOLD, 26, BLACK, "lt")
    y += name_h + 6
    _, _, _, ff_h = b.text(16, y, "AS / RUBY  ·  326224920", MED, 15, BLACK, "lt")
    y += ff_h + 6
    meta = "CONFIRM CTNGH3  ·  SEC 14  ·  TKT 027-2160918863  ·  20 KG"
    meta_w, _ = measure(meta, MED, 13)
    if meta_w > W - 32:
        raise SystemExit(f"meta line too wide: {meta_w:.0f}")
    _, _, _, meta_h = b.text(16, y, meta, MED, 13, BLACK, "lt")
    y += meta_h + 6

    status_h = 30
    status_top = content_bottom - status_h
    if y > status_top:
        raise SystemExit(f"content overlaps status: y={y} status_top={status_top}")
    # Red rail marks Google Flights data, which is not the airline gate field.
    b.rect(0, status_top, 8, status_h, RED)
    b.rect(8, status_top, W - 8, status_h, YELLOW)
    b.text(
        18,
        status_top + status_h / 2,
        "ON TIME  ·  GOOGLE GATE T12  ·  17:48 CST",
        BOLD,
        14,
        BLACK,
        "lm",
    )

    # QR, integer-scaled. Quiet zone is the surrounding white.
    qr = np.kron(mods, np.ones((QR_SCALE, QR_SCALE), dtype=np.uint8))
    black = qr == 1
    ys, xs = np.nonzero(black)
    b.arr[qr_y + ys, qr_x + xs] = BLACK

    if status_top + status_h > content_bottom:
        raise SystemExit("status enters the quiet zone")
    return b.image()


def preview(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    out = np.empty_like(arr)
    for src, dst in PIGMENT.items():
        out[(arr == src).all(axis=2)] = dst
    return Image.fromarray(out, "RGB")


def assert_pass(img: Image.Image, mods: np.ndarray) -> None:
    arr = np.array(img)
    if arr.shape != (H, W, 3):
        raise SystemExit(f"bad shape {arr.shape}")
    flat = arr.reshape(-1, 3)
    colors = {tuple(int(v) for v in c) for c in np.unique(flat, axis=0)}
    extra = colors - set(PALETTE)
    if extra:
        raise SystemExit(f"colors outside Spectra 6 palette: {extra}")

    qr_px = QR_MODULES * QR_SCALE
    quiet = QUIET_MODULES * QR_SCALE
    qr_x = (W - qr_px) // 2
    qr_y = H - quiet - qr_px

    crop = arr[qr_y : qr_y + qr_px, qr_x : qr_x + qr_px]
    for r in range(QR_MODULES):
        for c in range(QR_MODULES):
            block = crop[r * QR_SCALE : (r + 1) * QR_SCALE, c * QR_SCALE : (c + 1) * QR_SCALE]
            uniq = np.unique(block.reshape(-1, 3), axis=0)
            if len(uniq) != 1:
                raise SystemExit(f"module {r},{c} is not solid")
            got = tuple(int(v) for v in uniq[0])
            expect = BLACK if mods[r, c] else WHITE
            if got != expect:
                raise SystemExit(f"module {r},{c} is {got}, expected {expect}")

    zone = arr[qr_y - quiet : qr_y + qr_px + quiet, qr_x - quiet : qr_x + qr_px + quiet]
    # Everything in the quiet ring, outside the modules, must be white.
    ring = np.ones(zone.shape[:2], dtype=bool)
    ring[quiet : quiet + qr_px, quiet : quiet + qr_px] = False
    if not np.all(zone[ring] == WHITE):
        raise SystemExit("quiet zone is not white")

    import cv2

    det = cv2.QRCodeDetector()
    # Decode the QR crop plus its quiet zone so the rest of the pass cannot interfere.
    sample = zone.copy()
    value, _, _ = det.detectAndDecode(sample[:, :, ::-1])
    if value != PAYLOAD:
        raise SystemExit(f"QR decoded to {value!r}")


def main() -> None:
    mods = modules_from_hex()
    img = build(mods)
    assert_pass(img, mods)
    out_dir = "/workspace/00_Personal_Projects/boarding_pass"
    device = f"{out_dir}/boarding_pass.png"
    shown = f"{out_dir}/boarding_pass_preview.png"
    img.save(device, "PNG")
    preview(img).save(shown, "PNG")
    print(f"wrote {device}")
    print(f"wrote {shown}")
    print(f"size {img.size} scale {QR_SCALE}px/module")


if __name__ == "__main__":
    main()
