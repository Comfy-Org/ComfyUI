import numpy as np

from comfy_api.latest import ComfyExtension, io
from comfy_api.input import CurveInput
from typing_extensions import override


class CurveEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CurveEditor",
            display_name="Curve Editor",
            category="utilities",
            inputs=[
                io.Curve.Input("curve"),
                io.Histogram.Input("histogram", optional=True),
            ],
            outputs=[
                io.Curve.Output("curve"),
            ],
        )

    @classmethod
    def execute(cls, curve, histogram=None) -> io.NodeOutput:
        result = CurveInput.from_raw(curve)

        ui = {}
        if histogram is not None:
            ui["histogram"] = histogram if isinstance(histogram, list) else list(histogram)

        return io.NodeOutput(result, ui=ui) if ui else io.NodeOutput(result)


class ImageHistogram(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageHistogram",
            display_name="Image Histogram",
            category="utilities",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Histogram.Output("rgb"),
                io.Histogram.Output("luminance"),
                io.Histogram.Output("red"),
                io.Histogram.Output("green"),
                io.Histogram.Output("blue"),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        img = image[0].cpu().numpy()
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)

        def bincount(data):
            return np.bincount(data.ravel(), minlength=256)[:256]

        hist_r = bincount(img_uint8[:, :, 0])
        hist_g = bincount(img_uint8[:, :, 1])
        hist_b = bincount(img_uint8[:, :, 2])

        # Average of R, G, B histograms (same as Photoshop's RGB composite)
        rgb = ((hist_r + hist_g + hist_b) // 3).tolist()

        # ITU-R BT.709-6, Item 3.2 (p.6) — Derivation of luminance signal
        # https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        luminance = bincount(np.clip(lum * 255, 0, 255).astype(np.uint8)).tolist()

        return io.NodeOutput(
            rgb,
            luminance,
            hist_r.tolist(),
            hist_g.tolist(),
            hist_b.tolist(),
        )


class CurveExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [CurveEditor, ImageHistogram]


async def comfy_entrypoint():
    return CurveExtension()
