def hex_to_rgb(value: str) -> tuple[int, int, int]:
    h = value.lstrip("#")
    if len(h) != 6:
        return (255, 255, 255)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def readable_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum >= 130:
        return (r, g, b)
    t = (130 - lum) / (255 - lum)
    return (round(r + (255 - r) * t), round(g + (255 - g) * t), round(b + (255 - b) * t))


def normalize_palette(colors) -> list[str]:
    if isinstance(colors, dict):
        colors = colors.values()
    return [c.upper() for c in colors if isinstance(c, str) and c]
