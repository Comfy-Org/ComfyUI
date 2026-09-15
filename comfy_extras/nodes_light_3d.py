import math

import torch
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, IO
from comfy_extras.nodes_mesh_postprocess import (
    _any_hit_rays_bvh,
    _barycentric,
    _build_triangle_bvh,
    _camera_basis,
    _closest_hit_rays_bvh,
)

LIGHT_TYPES = ("directional", "point", "spot")

_DEFAULT_EYE = (0.0, 6.0, 8.0)
_DEFAULT_TARGET = (0.0, -0.5, 0.0)
_DEFAULT_FOV = 35.0
_DEFAULT_LIGHT_POSITION = (0.0, 1.8, 1.8)


def _hex_to_rgb01(color):
    s = str(color or "#ffffff").lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        return [1.0, 1.0, 1.0]


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hex_to_linear_rgb(color):
    return [_srgb_to_linear(c) for c in _hex_to_rgb01(color)]


def _linear_to_srgb(t):
    return torch.where(t <= 0.0031308, t * 12.92, 1.055 * t.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)


_SPHERE_ALBEDO = _srgb_to_linear(0.8)
_GROUND_ALBEDO = _srgb_to_linear(0.541)
_SKY_COLOR = '#8a8a8a'
_DEFAULT_AMBIENT_SKY = '#ffffff'
_DEFAULT_AMBIENT_GROUND = '#808080'


def _vec3(d, default=(0.0, 0.0, 0.0)):
    d = d or {}
    return [float(d.get(k, dv)) for k, dv in zip(("x", "y", "z"), default)]


def _uv_sphere(radius=1.0, center=(0.0, 0.0, 0.0), segments=48):
    rings, sectors = segments, segments
    theta = torch.linspace(0.0, math.pi, rings + 1)
    phi = torch.linspace(0.0, 2.0 * math.pi, sectors + 1)[:-1]
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    n = torch.stack([st[:, None] * sp[None, :], ct[:, None].expand(-1, sectors), st[:, None] * cp[None, :]], dim=-1)
    n = torch.nn.functional.normalize(n.reshape(-1, 3), dim=-1, eps=1e-6)
    v = n * radius + torch.tensor(center)
    idx = torch.arange((rings + 1) * sectors).reshape(rings + 1, sectors)
    nxt = torch.roll(idx, shifts=-1, dims=1)
    a, b = idx[:-1], idx[1:]
    c, d = nxt[:-1], nxt[1:]
    faces = torch.cat([torch.stack([a, b, d], dim=-1).reshape(-1, 3),
                       torch.stack([a, d, c], dim=-1).reshape(-1, 3)], dim=0)
    return v.float(), faces.long(), n.float()


def _ground_plane(size=100.0, y=-1.0):
    h = size * 0.5
    v = torch.tensor([[-h, y, -h], [-h, y, h], [h, y, h], [h, y, -h]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    n = torch.tensor([[0.0, 1.0, 0.0]] * 4, dtype=torch.float32)
    return v, faces, n


def _normalize_light(light):
    kind = str(light.get("type", "directional"))
    if kind not in LIGHT_TYPES:
        raise ValueError(f"Unsupported light type '{kind}'. Supported: {', '.join(LIGHT_TYPES)}")
    return {
        "type": kind,
        "color": _hex_to_linear_rgb(light.get("color")),
        "intensity": max(0.0, float(light.get("intensity", 1.0))),
        "position": _vec3(light.get("position"), _DEFAULT_LIGHT_POSITION),
        "target": _vec3(light.get("target")),
        "range": max(0.0, float(light.get("range", 0.0) or 0.0)),
        "inner_cone": float(light.get("innerConeAngle", 30.0)),
        "outer_cone": float(light.get("outerConeAngle", 45.0)),
        "radius": max(0.0, float(light.get("radius", 0.0) or 0.0)),
        "cast_shadow": bool(light.get("castShadow", True)),
    }


def _clean_light(light) -> IO.Load3DLightInfo.LightInfo:
    kind = str(light.get("type", "directional"))
    if kind not in LIGHT_TYPES:
        raise ValueError(f"Unsupported light type '{kind}'. Supported: {', '.join(LIGHT_TYPES)}")
    x, y, z = _vec3(light.get("position"), _DEFAULT_LIGHT_POSITION)
    cleaned: IO.Load3DLightInfo.LightInfo = {
        "type": kind,
        "color": str(light.get("color", "#ffffff")),
        "intensity": max(0.0, float(light.get("intensity", 1.0))),
        "position": {"x": x, "y": y, "z": z},
    }
    if kind != "point":
        tx, ty, tz = _vec3(light.get("target"))
        cleaned["target"] = {"x": tx, "y": ty, "z": tz}
    if kind != "directional":
        rng = float(light.get("range", 0.0) or 0.0)
        if rng > 0.0:
            cleaned["range"] = rng
    if kind == "spot":
        outer = float(light.get("outerConeAngle", 45.0))
        cleaned["innerConeAngle"] = min(float(light.get("innerConeAngle", 30.0)), outer)
        cleaned["outerConeAngle"] = outer
    radius = float(light.get("radius", 0.0) or 0.0)
    if radius > 0.0:
        cleaned["radius"] = radius
    if light.get("castShadow", True) is False:
        cleaned["castShadow"] = False
    return cleaned


def _vogel_disk(n, dev):
    i = torch.arange(n, device=dev, dtype=torch.float32) + 0.5
    r = torch.sqrt(i / n)
    theta = i * 2.399963229728653
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


def _per_point_rotation(P):
    h = torch.sin((P * torch.tensor([12.9898, 78.233, 37.719], device=P.device)).sum(-1)) * 43758.5453
    return (h - h.floor()) * (2.0 * math.pi)


def _tangent_frame(L):
    up = torch.tensor([0.0, 1.0, 0.0], device=L.device).expand_as(L)
    side = torch.tensor([1.0, 0.0, 0.0], device=L.device).expand_as(L)
    helper = torch.where(L[:, 1:2].abs() < 0.9, up, side)
    t1 = torch.nn.functional.normalize(torch.cross(L, helper, dim=-1), dim=-1, eps=1e-6)
    t2 = torch.cross(L, t1, dim=-1)
    return t1, t2


def _shadow_fraction(light, P, N, L, dist, tri, bvh, samples):
    dev = P.device
    origin = P + N * 1e-3
    radius = light["radius"]
    if radius <= 0.0 or samples <= 1:
        tmax = torch.full((P.shape[0],), 1e30, device=dev) if dist is None else dist - 1e-3
        return _any_hit_rays_bvh(origin, L, tri, bvh, tmin=1e-4, tmax=tmax).float()
    t1, t2 = _tangent_frame(L)
    phi = _per_point_rotation(P)
    cos_phi, sin_phi = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
    blocked = torch.zeros(P.shape[0], device=dev)
    for ox, oy in _vogel_disk(samples, dev).tolist():
        u = ox * cos_phi - oy * sin_phi
        v = ox * sin_phi + oy * cos_phi
        offset = t1 * u + t2 * v
        if dist is None:
            spread = math.tan(math.radians(radius) * 0.5)
            dirs = torch.nn.functional.normalize(L + offset * spread, dim=-1, eps=1e-6)
            tmax = torch.full((P.shape[0],), 1e30, device=dev)
        else:
            to_sample = L * dist[:, None] + offset * radius
            tmax = to_sample.norm(dim=-1) - 1e-3
            dirs = to_sample / tmax.clamp_min(1e-6)[:, None]
        blocked += _any_hit_rays_bvh(origin, dirs, tri, bvh, tmin=1e-4, tmax=tmax).float()
    return blocked / samples


def _light_contribution(light, P, N, tri, bvh, dev, shadow_samples):
    color = torch.tensor(light["color"], device=dev) * light["intensity"]
    pos = torch.tensor(light["position"], device=dev)
    if light["type"] == "directional":
        L = torch.nn.functional.normalize(
            pos - torch.tensor(light["target"], device=dev), dim=-1, eps=1e-6).expand_as(P)
        atten = torch.ones(P.shape[0], device=dev)
        dist = None
    else:
        to_light = pos[None, :] - P
        dist = to_light.norm(dim=-1).clamp_min(1e-4)
        L = to_light / dist[:, None]
        atten = 1.0 / dist.square().clamp_min(0.01)
        if light["range"] > 0.0:
            win = (1.0 - (dist / light["range"]).pow(4)).clamp(0.0, 1.0)
            atten = atten * win.square()
        if light["type"] == "spot":
            aim = torch.nn.functional.normalize(
                torch.tensor(light["target"], device=dev) - pos, dim=-1, eps=1e-6)
            cos_outer = math.cos(math.radians(max(light["outer_cone"], 1e-3)))
            cos_inner = math.cos(math.radians(min(light["inner_cone"], light["outer_cone"])))
            cos_dir = (-L * aim[None, :]).sum(-1)
            t = ((cos_dir - cos_outer) / max(cos_inner - cos_outer, 1e-6)).clamp(0.0, 1.0)
            atten = atten * (t * t * (3.0 - 2.0 * t))
    ndl = (N * L).sum(-1).clamp_min(0.0)
    lit = (ndl > 0.0) & (atten > 0.0)
    visible = torch.ones(P.shape[0], device=dev)
    if light["cast_shadow"] and bool(lit.any()):
        visible[lit] = 1.0 - _shadow_fraction(
            light, P[lit], N[lit], L[lit], None if dist is None else dist[lit], tri, bvh,
            shadow_samples)
    factor = ndl * atten * visible
    return color[None, :] * factor[:, None]


class CreateLightInfo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreateLightInfo",
            display_name="Create Light Info",
            search_aliases=["light position", "make light info", "directional light",
                            "point light", "spot light"],
            category="3d/light",
            is_experimental=True,
            description="Edit one or more lights (directional, point, spot) in an interactive "
                        "3D studio preview. Coordinates are the viewer's world space "
                        "(right-handed, Y-up).",
            inputs=[
                IO.LightInfoPreview.Input(
                    "editor_state",
                    tooltip="Lights edited in the interactive 3D preview."),
            ],
            outputs=[IO.Load3DLightInfo.Output(display_name="light_info")],
        )

    @classmethod
    def execute(cls, editor_state=None) -> IO.NodeOutput:
        return IO.NodeOutput([_clean_light(light) for light in (editor_state or [])
                              if isinstance(light, dict)])


class RenderLight(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RenderLight",
            display_name="Render Light",
            search_aliases=["light preview", "sphere light render", "render lighting",
                            "lighting reference"],
            category="3d/light",
            is_experimental=True,
            description="Ray-casts a studio scene (clay sphere on a ground plane) lit by the "
                        "connected light_info: Lambert shading with soft shadows, distance "
                        "falloff, and spot cones, all lights accumulated, matching the Create "
                        "Light Info preview. Also outputs one image per light. Useful as a "
                        "lighting reference. The camera comes from an optional camera_info.",
            inputs=[
                IO.Load3DLightInfo.Input("light_info",
                                         tooltip="One or more lights from Create Light Info nodes."),
                IO.Int.Input("width", default=1024, min=64, max=4096, step=8),
                IO.Int.Input("height", default=1024, min=64, max=4096, step=8),
                IO.Float.Input("ambient", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="Hemisphere fill light strength so unlit areas are not "
                                       "pitch black. Surfaces facing up get sky_color, facing "
                                       "down get ground_color."),
                IO.Color.Input("sky_color", default=_DEFAULT_AMBIENT_SKY,
                               tooltip="Fill light color from above."),
                IO.Color.Input("ground_color", default=_DEFAULT_AMBIENT_GROUND,
                               tooltip="Fill light color from below."),
                IO.Int.Input("shadow_samples", default=16, min=1, max=64,
                             tooltip="Shadow rays per pixel for lights with a radius (soft shadows). "
                                     "1 = hard shadows."),
                IO.Load3DCamera.Input("camera_info", optional=True,
                                      tooltip="Camera from a Load3D / Preview3D viewer or a "
                                              "Create Camera Info node. If none is connected, "
                                              "the default studio view is used."),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Image.Output(display_name="per_light",
                                tooltip="One image per light, in light_info order, with only "
                                        "that light's contribution and no ambient."),
            ],
        )

    @classmethod
    def execute(cls, light_info, width, height, ambient, sky_color=_DEFAULT_AMBIENT_SKY,
                ground_color=_DEFAULT_AMBIENT_GROUND, shadow_samples=16,
                camera_info=None) -> IO.NodeOutput:
        dev = comfy.model_management.get_torch_device()
        lights = [_normalize_light(light) for light in (light_info or [])]

        sv, sf, sn = _uv_sphere()
        gv, gf, gn = _ground_plane()
        verts = torch.cat([sv, gv]).to(dev)
        faces = torch.cat([sf, gf + sv.shape[0]]).to(dev)
        normals = torch.cat([sn, gn]).to(dev)
        sphere_face_count = sf.shape[0]
        albedo_per_face = torch.where(
            torch.arange(faces.shape[0], device=dev) < sphere_face_count,
            _SPHERE_ALBEDO, _GROUND_ALBEDO)

        tri = verts[faces]
        bvh = _build_triangle_bvh(tri)

        up_hint = torch.tensor([0.0, 1.0, 0.0], device=dev)
        if camera_info is not None:
            eye = torch.tensor(_vec3(camera_info.get("position"), _DEFAULT_EYE), device=dev)
            target = torch.tensor(_vec3(camera_info.get("target"), _DEFAULT_TARGET), device=dev)
            cam_fov = float(camera_info.get("fov", 0) or _DEFAULT_FOV)
            cam_zoom = float(camera_info.get("zoom", 1.0) or 1.0)
            fov_rad = 2.0 * math.atan(math.tan(math.radians(cam_fov) * 0.5) / cam_zoom)
        else:
            eye = torch.tensor(_DEFAULT_EYE, device=dev)
            target = torch.tensor(_DEFAULT_TARGET, device=dev)
            fov_rad = math.radians(_DEFAULT_FOV)
        f, r, u = _camera_basis(eye, target, up_hint)

        H, W = int(height), int(width)
        ys = 1.0 - (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H * 2.0
        xs = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W * 2.0 - 1.0
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        tn = math.tan(0.5 * fov_rad)
        aspect = W / H
        d = torch.nn.functional.normalize(
            (r * (gx * tn * aspect)[..., None] + u * (gy * tn)[..., None] + f).reshape(-1, 3),
            dim=-1, eps=1e-6)
        o = eye[None, :].expand(H * W, 3)

        bg_rgb = torch.tensor(_hex_to_rgb01(_SKY_COLOR), device=dev)
        sky_rgb = torch.tensor(_hex_to_linear_rgb(sky_color), device=dev) * float(ambient)
        ground_rgb = torch.tensor(_hex_to_linear_rgb(ground_color), device=dev) * float(ambient)
        img = bg_rgb[None, :].repeat(H * W, 1)
        per_light = [bg_rgb[None, :].repeat(H * W, 1) for _ in lights]
        ray_chunk = 1 << 22
        for s in range(0, H * W, ray_chunk):
            e = min(s + ray_chunk, H * W)
            t_hit, face, hit = _closest_hit_rays_bvh(o[s:e], d[s:e], tri, bvh, tmin=1e-5, tmax=1e30)
            if not bool(hit.any()):
                continue
            fh = face[hit].clamp_min(0)
            P = o[s:e][hit] + t_hit[hit, None] * d[s:e][hit]
            bary = _barycentric(P, tri[fh])
            N = torch.nn.functional.normalize(
                (bary[:, :, None] * normals[faces[fh]]).sum(1), dim=-1, eps=1e-6)
            albedo = albedo_per_face[fh, None] / math.pi
            up_mix = (N[:, 1:2] * 0.5 + 0.5)
            radiance = ground_rgb[None, :] + (sky_rgb - ground_rgb)[None, :] * up_mix
            for light, buf in zip(lights, per_light):
                contribution = _light_contribution(light, P, N, tri, bvh, dev, shadow_samples)
                radiance = radiance + contribution
                local = buf[s:e]
                local[hit] = _linear_to_srgb(albedo * contribution).clamp(0.0, 1.0)
                buf[s:e] = local
            local = img[s:e]
            local[hit] = _linear_to_srgb(albedo * radiance).clamp(0.0, 1.0)
            img[s:e] = local

        image = img.reshape(H, W, 3)[None].cpu()
        per_light_images = (torch.stack([buf.reshape(H, W, 3) for buf in per_light]).cpu()
                            if per_light else image)
        return IO.NodeOutput(image, per_light_images)


class Light3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [CreateLightInfo, RenderLight]


async def comfy_entrypoint() -> Light3DExtension:
    return Light3DExtension()
