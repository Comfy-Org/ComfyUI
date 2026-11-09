from io import BytesIO


class SVG:
    """Stores SVG representations via a list of BytesIO objects."""

    def __init__(self, data: list[BytesIO]):
        self.data = data

    def combine(self, other: 'SVG') -> 'SVG':
        return SVG(self.data + other.data)

    @staticmethod
    def combine_all(svgs: list['SVG']) -> 'SVG':
        all_svgs_list: list[BytesIO] = []
        for svg_item in svgs:
            all_svgs_list.extend(svg_item.data)
        return SVG(all_svgs_list)
