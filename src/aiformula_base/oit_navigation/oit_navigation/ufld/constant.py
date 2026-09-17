# Row anchors from the official Ultra-Fast-Lane-Detection repo (cfzd/Ultra-Fast-Lane-Detection).
# Defined in the network's canonical 288-pixel-tall input space; scaled to the real image
# height at decode time. TuSimple uses 56 row anchors (backbone trained on TuSimple).
TUSIMPLE_ROW_ANCHOR = [
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
    116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
    168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
    220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
    272, 276, 280, 284,
]

TUSIMPLE_GRIDING_NUM = 100
TUSIMPLE_NUM_LANES = 4
TUSIMPLE_INPUT_SIZE = (800, 288)  # (width, height)
