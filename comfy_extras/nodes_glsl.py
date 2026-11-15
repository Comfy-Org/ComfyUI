import os
import sys
import re
import ctypes
import logging
from typing import TypedDict

import numpy as np
import torch

import nodes
import comfy_angle
from comfy_api.latest import ComfyExtension, io, ui
from typing_extensions import override

logger = logging.getLogger(__name__)


def _preload_angle():
    egl_path = comfy_angle.get_egl_path()
    gles_path = comfy_angle.get_glesv2_path()

    if sys.platform == "win32":
        angle_dir = comfy_angle.get_lib_dir()
        os.add_dll_directory(angle_dir)
        os.environ["PATH"] = angle_dir + os.pathsep + os.environ.get("PATH", "")

    mode = 0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL
    ctypes.CDLL(str(egl_path), mode=mode)
    ctypes.CDLL(str(gles_path), mode=mode)


# Pre-load ANGLE *before* any PyOpenGL import so that the EGL platform
# plugin picks up ANGLE's libEGL / libGLESv2 instead of system libs.
_preload_angle()
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


import OpenGL
OpenGL.USE_ACCELERATE = False


def _patch_find_library():
    """PyOpenGL's EGL platform looks for 'EGL' and 'GLESv2' by short name
    via ctypes.util.find_library, but ANGLE ships as 'libEGL' and
    'libGLESv2'.  Patch find_library to return the full ANGLE paths so
    PyOpenGL loads the same libraries we pre-loaded."""
    if sys.platform == "linux":
        return
    import ctypes.util
    _orig = ctypes.util.find_library
    def _patched(name):
        if name == 'EGL':
            return comfy_angle.get_egl_path()
        if name == 'GLESv2':
            return comfy_angle.get_glesv2_path()
        return _orig(name)
    ctypes.util.find_library = _patched


_patch_find_library()

from OpenGL import EGL
from OpenGL import GLES3 as gl

class SizeModeInput(TypedDict):
    size_mode: str
    width: int
    height: int


MAX_IMAGES = 5      # u_image0-4
MAX_UNIFORMS = 20   # u_float0-19, u_int0-19
MAX_BOOLS = 10      # u_bool0-9
MAX_CURVES = 4      # u_curve0-3 (1D LUT textures)
MAX_OUTPUTS = 4     # fragColor0-3 (MRT)

# Vertex shader using gl_VertexID trick - no VBO needed.
# Draws a single triangle that covers the entire screen:
#
#     (-1,3)
#       /|
#      / |  <- visible area is the unit square from (-1,-1) to (1,1)
#     /  |     parts outside get clipped away
# (-1,-1)---(3,-1)
#
# v_texCoord is computed from clip space: * 0.5 + 0.5 maps (-1,1) -> (0,1)
VERTEX_SHADER = """#version 300 es
out vec2 v_texCoord;
void main() {
    vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    v_texCoord = verts[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
}
"""

DEFAULT_FRAGMENT_SHADER = """#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform vec2 u_resolution;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"""



def _egl_attribs(*values):
    """Build an EGL_NONE-terminated EGLint attribute array."""
    vals = list(values) + [EGL.EGL_NONE]
    return (ctypes.c_int32 * len(vals))(*vals)


# EGL platform extension constants
EGL_PLATFORM_ANGLE_ANGLE = 0x3202
EGL_PLATFORM_ANGLE_TYPE_ANGLE = 0x3203
EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE = 0x3450
EGL_MESA_PLATFORM_SURFACELESS = 0x31DD


_eglGetPlatformDisplayEXT = None

def _get_egl_platform_display_ext(platform, native_display, attribs):
    """Call eglGetPlatformDisplayEXT via ctypes (extension, not in PyOpenGL)."""
    global _eglGetPlatformDisplayEXT
    if _eglGetPlatformDisplayEXT is None:
        from OpenGL import platform as _plat
        egl_lib = _plat.PLATFORM.EGL
        _get_proc = egl_lib.eglGetProcAddress
        _get_proc.restype = ctypes.c_void_p
        _get_proc.argtypes = [ctypes.c_char_p]
        ptr = _get_proc(b"eglGetPlatformDisplayEXT")
        if not ptr:
            return None
        func_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p)
        _eglGetPlatformDisplayEXT = func_type(ptr)

    raw = _eglGetPlatformDisplayEXT(platform, native_display, attribs)
    if not raw:
        return None
    return ctypes.cast(raw, EGL.EGLDisplay)


def _get_egl_display():
    """Get an EGL display, trying the default first then ANGLE's Vulkan
    platform for headless environments without a display server."""
    failures = []

    # Try the default display first (works when X11/Wayland is available)
    display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if display:
        major, minor = ctypes.c_int32(0), ctypes.c_int32(0)
        try:
            if EGL.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
                return display, major.value, minor.value
        except Exception as e:
            failures.append(f"default: {e}")

    logger.info("Default EGL display unavailable, trying headless fallbacks")

    # Headless fallback strategies, tried in order:
    headless_strategies = [
        ("surfaceless", EGL_MESA_PLATFORM_SURFACELESS, None, None),
        ("ANGLE Vulkan", EGL_PLATFORM_ANGLE_ANGLE, None,
         _egl_attribs(EGL_PLATFORM_ANGLE_TYPE_ANGLE, EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE)),
    ]

    for name, platform, native_display, attribs in headless_strategies:
        display = _get_egl_platform_display_ext(platform, native_display, attribs)
        if not display:
            failures.append(f"{name}: eglGetPlatformDisplayEXT returned no display")
            continue
        major, minor = ctypes.c_int32(0), ctypes.c_int32(0)
        try:
            if EGL.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
                logger.info(f"Using EGL {name} platform (headless)")
                return display, major.value, minor.value
            failures.append(f"{name}: eglInitialize returned false")
        except Exception as e:
            failures.append(f"{name}: {e}")
            continue

    details = "\n".join(f"  - {f}" for f in failures)
    raise RuntimeError(
        "Failed to initialize EGL display.\n"
        "No display server and no headless EGL platform available.\n"
        f"Tried:\n{details}\n"
        "Ensure GPU drivers are installed or set DISPLAY for a virtual framebuffer."
    )


def _gl_str(name):
    """Get an OpenGL string parameter."""
    v = gl.glGetString(name)
    if not v:
        return "Unknown"
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return ctypes.string_at(v).decode(errors="replace")


def _detect_output_count(source: str) -> int:
    """Detect how many fragColor outputs are used in the shader.

    Returns the count of outputs needed (1 to MAX_OUTPUTS).
    """
    matches = re.findall(r"fragColor(\d+)", source)
    if not matches:
        return 1  # Default to 1 output if none found
    max_index = max(int(m) for m in matches)
    return min(max_index + 1, MAX_OUTPUTS)


def _detect_pass_count(source: str) -> int:
    """Detect multi-pass rendering from #pragma passes N directive.

    Returns the number of passes (1 if not specified).
    """
    match = re.search(r'#pragma\s+passes\s+(\d+)', source)
    if match:
        return max(1, int(match.group(1)))
    return 1


class GLContext:
    """Manages an OpenGL ES 3.0 context via EGL/ANGLE (singleton)."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if GLContext._initialized:
            return

        import time
        start = time.perf_counter()

        self._display = None
        self._surface = None
        self._context = None
        self._vao = None

        try:
            self._display, self._egl_major, self._egl_minor = _get_egl_display()

            if not EGL.eglBindAPI(EGL.EGL_OPENGL_ES_API):
                raise RuntimeError("eglBindAPI(EGL_OPENGL_ES_API) failed")

            config = EGL.EGLConfig()
            n_configs = ctypes.c_int32(0)
            if not EGL.eglChooseConfig(
                self._display,
                _egl_attribs(
                    EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_ES3_BIT,
                    EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                    EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8,
                    EGL.EGL_BLUE_SIZE, 8, EGL.EGL_ALPHA_SIZE, 8,
                ),
                ctypes.byref(config), 1, ctypes.byref(n_configs),
            ) or n_configs.value == 0:
                raise RuntimeError("eglChooseConfig() failed")

            self._surface = EGL.eglCreatePbufferSurface(
                self._display, config,
                _egl_attribs(EGL.EGL_WIDTH, 64, EGL.EGL_HEIGHT, 64),
            )
            if not self._surface:
                raise RuntimeError("eglCreatePbufferSurface() failed")

            self._context = EGL.eglCreateContext(
                self._display, config, EGL.EGL_NO_CONTEXT,
                _egl_attribs(EGL.EGL_CONTEXT_CLIENT_VERSION, 3),
            )
            if not self._context:
                raise RuntimeError("eglCreateContext() failed")

            if not EGL.eglMakeCurrent(self._display, self._surface, self._surface, self._context):
                raise RuntimeError("eglMakeCurrent() failed")

            self._vao = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(self._vao)

        except Exception:
            self._cleanup()
            raise

        elapsed = (time.perf_counter() - start) * 1000

        renderer = _gl_str(gl.GL_RENDERER)
        vendor = _gl_str(gl.GL_VENDOR)
        version = _gl_str(gl.GL_VERSION)

        GLContext._initialized = True
        logger.info(f"GLSL context initialized in {elapsed:.1f}ms - EGL {self._egl_major}.{self._egl_minor}, {renderer} ({vendor}), GL {version}")

    def make_current(self):
        if not EGL.eglMakeCurrent(self._display, self._surface, self._surface, self._context):
            err = EGL.eglGetError()
            raise RuntimeError(f"eglMakeCurrent() failed (EGL error: 0x{err:04X})")
        if self._vao is not None:
            gl.glBindVertexArray(self._vao)

    def _cleanup(self):
        if not self._display:
            return
        try:
            if self._vao is not None:
                gl.glDeleteVertexArrays(1, [self._vao])
                self._vao = None
        except Exception:
            pass
        try:
            EGL.eglMakeCurrent(self._display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
        except Exception:
            pass
        try:
            if self._context:
                EGL.eglDestroyContext(self._display, self._context)
        except Exception:
            pass
        try:
            if self._surface:
                EGL.eglDestroySurface(self._display, self._surface)
        except Exception:
            pass
        try:
            EGL.eglTerminate(self._display)
        except Exception:
            pass
        self._display = None


def _compile_shader(source: str, shader_type: int) -> int:
    """Compile a shader and return its ID."""
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader)
        if isinstance(error, bytes):
            error = error.decode(errors="replace")
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation failed:\n{error}")

    return shader


def _create_program(vertex_source: str, fragment_source: str) -> int:
    """Create and link a shader program."""
    vertex_shader = _compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
    try:
        fragment_shader = _compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
    except RuntimeError:
        gl.glDeleteShader(vertex_shader)
        raise

    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)

    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        error = gl.glGetProgramInfoLog(program)
        if isinstance(error, bytes):
            error = error.decode(errors="replace")
        gl.glDeleteProgram(program)
        raise RuntimeError(f"Program linking failed:\n{error}")

    return program


def _render_shader_batch(
    fragment_code: str,
    width: int,
    height: int,
    image_batches: list[list[np.ndarray]],
    floats: list[float],
    ints: list[int],
    bools: list[bool] | None = None,
    curves: list[np.ndarray] | None = None,
) -> list[list[np.ndarray]]:
    """
    Render a fragment shader for multiple batches efficiently.

    Compiles shader once, reuses framebuffer/textures across batches.
    Supports multi-pass rendering via #pragma passes N directive.

    Args:
        fragment_code: User's fragment shader code
        width: Output width
        height: Output height
        image_batches: List of batches, each batch is a list of input images (H, W, C) float32 [0,1]
        floats: List of float uniforms
        ints: List of int uniforms
        bools: List of bool uniforms (passed as int 0/1 to GLSL bool uniforms)
        curves: List of 1D LUT arrays (float32) of arbitrary size for u_curve0-N

    Returns:
        List of batch outputs, each is a list of output images (H, W, 4) float32 [0,1]
    """
    import time
    start_time = time.perf_counter()

    if not image_batches:
        return []

    ctx = GLContext()
    ctx.make_current()

    # Detect how many outputs the shader actually uses
    num_outputs = _detect_output_count(fragment_code)

    # Detect multi-pass rendering
    num_passes = _detect_pass_count(fragment_code)

    if bools is None:
        bools = []
    if curves is None:
        curves = []

    # Track resources for cleanup
    program = None
    fbo = None
    output_textures = []
    input_textures = []
    curve_textures = []
    ping_pong_textures = []
    ping_pong_fbos = []

    num_inputs = len(image_batches[0])

    try:
        # Compile shaders (once for all batches)
        try:
            program = _create_program(VERTEX_SHADER, fragment_code)
        except RuntimeError:
            logger.error(f"Fragment shader:\n{fragment_code}")
            raise

        gl.glUseProgram(program)

        # Create framebuffer with only the needed color attachments
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        draw_buffers = []
        for i in range(num_outputs):
            tex = gl.glGenTextures(1)
            output_textures.append(tex)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + i, gl.GL_TEXTURE_2D, tex, 0)
            draw_buffers.append(gl.GL_COLOR_ATTACHMENT0 + i)

        gl.glDrawBuffers(num_outputs, draw_buffers)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete")

        # Create ping-pong resources for multi-pass rendering
        if num_passes > 1:
            for _ in range(2):
                pp_tex = gl.glGenTextures(1)
                ping_pong_textures.append(pp_tex)
                gl.glBindTexture(gl.GL_TEXTURE_2D, pp_tex)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

                pp_fbo = gl.glGenFramebuffers(1)
                ping_pong_fbos.append(pp_fbo)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, pp_fbo)
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, pp_tex, 0)
                gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])

                if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError("Ping-pong framebuffer is not complete")

        # Create input textures (reused for all batches)
        for i in range(num_inputs):
            tex = gl.glGenTextures(1)
            input_textures.append(tex)
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            loc = gl.glGetUniformLocation(program, f"u_image{i}")
            if loc >= 0:
                gl.glUniform1i(loc, i)

        # Set static uniforms (once for all batches)
        loc = gl.glGetUniformLocation(program, "u_resolution")
        if loc >= 0:
            gl.glUniform2f(loc, float(width), float(height))

        for i, v in enumerate(floats):
            loc = gl.glGetUniformLocation(program, f"u_float{i}")
            if loc >= 0:
                gl.glUniform1f(loc, v)

        for i, v in enumerate(ints):
            loc = gl.glGetUniformLocation(program, f"u_int{i}")
            if loc >= 0:
                gl.glUniform1i(loc, v)

        for i, v in enumerate(bools):
            loc = gl.glGetUniformLocation(program, f"u_bool{i}")
            if loc >= 0:
                gl.glUniform1i(loc, 1 if v else 0)

        # Create 1D LUT textures for curves (bound after image texture units)
        for i, lut in enumerate(curves):
            tex = gl.glGenTextures(1)
            curve_textures.append(tex)
            unit = MAX_IMAGES + i
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, len(lut), 1, 0, gl.GL_RED, gl.GL_FLOAT, lut)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            loc = gl.glGetUniformLocation(program, f"u_curve{i}")
            if loc >= 0:
                gl.glUniform1i(loc, unit)

        # Get u_pass uniform location for multi-pass
        pass_loc = gl.glGetUniformLocation(program, "u_pass")

        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_BLEND)  # Ensure no alpha blending - write output directly

        # Process each batch
        all_batch_outputs = []
        for images in image_batches:
            # Update input textures with this batch's images
            for i, img in enumerate(images):
                gl.glActiveTexture(gl.GL_TEXTURE0 + i)
                gl.glBindTexture(gl.GL_TEXTURE_2D, input_textures[i])

                # Flip vertically for GL coordinates, ensure RGBA
                h, w, c = img.shape
                if c == 3:
                    img_upload = np.empty((h, w, 4), dtype=np.float32)
                    img_upload[:, :, :3] = img[::-1, :, :]
                    img_upload[:, :, 3] = 1.0
                else:
                    img_upload = np.ascontiguousarray(img[::-1, :, :])

                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, img_upload)

            if num_passes == 1:
                # Single pass - render directly to output FBO
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
                if pass_loc >= 0:
                    gl.glUniform1i(pass_loc, 0)
                gl.glClearColor(0, 0, 0, 0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            else:
                # Multi-pass rendering with ping-pong
                for p in range(num_passes):
                    is_last_pass = (p == num_passes - 1)

                    # Set pass uniform
                    if pass_loc >= 0:
                        gl.glUniform1i(pass_loc, p)

                    if is_last_pass:
                        # Last pass renders to the main output FBO
                        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
                    else:
                        # Intermediate passes render to ping-pong FBO
                        target_fbo = ping_pong_fbos[p % 2]
                        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target_fbo)

                    # Set input texture for this pass
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    if p == 0:
                        # First pass reads from original input
                        gl.glBindTexture(gl.GL_TEXTURE_2D, input_textures[0])
                    else:
                        # Subsequent passes read from previous pass output
                        source_tex = ping_pong_textures[(p - 1) % 2]
                        gl.glBindTexture(gl.GL_TEXTURE_2D, source_tex)

                    gl.glClearColor(0, 0, 0, 0)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

            # Read back outputs for this batch
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            batch_outputs = []
            for i in range(num_outputs):
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                buf = np.empty((height, width, 4), dtype=np.float32)
                gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_FLOAT, buf)
                batch_outputs.append(buf[::-1, :, :].copy())

            # Pad with black images for unused outputs
            black_img = np.zeros((height, width, 4), dtype=np.float32)
            for _ in range(num_outputs, MAX_OUTPUTS):
                batch_outputs.append(black_img)

            all_batch_outputs.append(batch_outputs)

        elapsed = (time.perf_counter() - start_time) * 1000
        num_batches = len(image_batches)
        pass_info = f", {num_passes} passes" if num_passes > 1 else ""
        logger.info(f"GLSL shader executed in {elapsed:.1f}ms ({num_batches} batch{'es' if num_batches != 1 else ''}, {width}x{height}{pass_info})")

        return all_batch_outputs

    finally:
        # Unbind before deleting
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

        if input_textures:
            gl.glDeleteTextures(len(input_textures), input_textures)
        if curve_textures:
            gl.glDeleteTextures(len(curve_textures), curve_textures)
        if output_textures:
            gl.glDeleteTextures(len(output_textures), output_textures)
        if ping_pong_textures:
            gl.glDeleteTextures(len(ping_pong_textures), ping_pong_textures)
        if fbo is not None:
            gl.glDeleteFramebuffers(1, [fbo])
        if ping_pong_fbos:
            gl.glDeleteFramebuffers(len(ping_pong_fbos), ping_pong_fbos)
        if program is not None:
            gl.glDeleteProgram(program)

class GLSLShader(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"),
            prefix="image",
            min=1,
            max=MAX_IMAGES,
        )

        float_template = io.Autogrow.TemplatePrefix(
            io.Float.Input("float", default=0.0),
            prefix="u_float",
            min=0,
            max=MAX_UNIFORMS,
        )

        int_template = io.Autogrow.TemplatePrefix(
            io.Int.Input("int", default=0),
            prefix="u_int",
            min=0,
            max=MAX_UNIFORMS,
        )

        bool_template = io.Autogrow.TemplatePrefix(
            io.Boolean.Input("bool", default=False),
            prefix="u_bool",
            min=0,
            max=MAX_BOOLS,
        )

        curve_template = io.Autogrow.TemplatePrefix(
            io.Curve.Input("curve"),
            prefix="u_curve",
            min=0,
            max=MAX_CURVES,
        )

        return io.Schema(
            node_id="GLSLShader",
            display_name="GLSL Shader",
            category="image/shader",
            description=(
                "Apply GLSL ES fragment shaders to images. "
                "u_resolution (vec2) is always available."
            ),
            is_experimental=True,
            has_intermediate_output=True,
            inputs=[
                io.String.Input(
                    "fragment_shader",
                    default=DEFAULT_FRAGMENT_SHADER,
                    multiline=True,
                    tooltip="GLSL fragment shader source code (GLSL ES 3.00 / WebGL 2.0 compatible)",
                ),
                io.DynamicCombo.Input(
                    "size_mode",
                    options=[
                        io.DynamicCombo.Option("from_input", []),
                        io.DynamicCombo.Option(
                            "custom",
                            [
                                io.Int.Input(
                                    "width",
                                    default=512,
                                    min=1,
                                    max=nodes.MAX_RESOLUTION,
                                ),
                                io.Int.Input(
                                    "height",
                                    default=512,
                                    min=1,
                                    max=nodes.MAX_RESOLUTION,
                                ),
                            ],
                        ),
                    ],
                    tooltip="Output size: 'from_input' uses first input image dimensions, 'custom' allows manual size",
                ),
                io.Autogrow.Input("images", template=image_template, tooltip=f"Images are available as u_image0-{MAX_IMAGES-1} (sampler2D) in the shader code"),
                io.Autogrow.Input("floats", template=float_template, tooltip=f"Floats are available as u_float0-{MAX_UNIFORMS-1} in the shader code"),
                io.Autogrow.Input("ints", template=int_template, tooltip=f"Ints are available as u_int0-{MAX_UNIFORMS-1} in the shader code"),
                io.Autogrow.Input("bools", template=bool_template, tooltip=f"Booleans are available as u_bool0-{MAX_BOOLS-1} (bool) in the shader code"),
                io.Autogrow.Input("curves", template=curve_template, tooltip=f"Curves are available as u_curve0-{MAX_CURVES-1} (sampler2D, 1D LUT) in the shader code. Sample with texture(u_curve0, vec2(x, 0.5)).r"),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE0", tooltip="Available via layout(location = 0) out vec4 fragColor0 in the shader code"),
                io.Image.Output(display_name="IMAGE1", tooltip="Available via layout(location = 1) out vec4 fragColor1 in the shader code"),
                io.Image.Output(display_name="IMAGE2", tooltip="Available via layout(location = 2) out vec4 fragColor2 in the shader code"),
                io.Image.Output(display_name="IMAGE3", tooltip="Available via layout(location = 3) out vec4 fragColor3 in the shader code"),
            ],
        )

    @classmethod
    def execute(
        cls,
        fragment_shader: str,
        size_mode: SizeModeInput,
        images: io.Autogrow.Type,
        floats: io.Autogrow.Type = None,
        ints: io.Autogrow.Type = None,
        bools: io.Autogrow.Type = None,
        curves: io.Autogrow.Type = None,
        **kwargs,
    ) -> io.NodeOutput:

        image_list = [v for v in images.values() if v is not None]
        float_list = (
            [v if v is not None else 0.0 for v in floats.values()] if floats else []
        )
        int_list = [v if v is not None else 0 for v in ints.values()] if ints else []
        bool_list = [v if v is not None else False for v in bools.values()] if bools else []

        curve_luts = [v.to_lut().astype(np.float32) for v in curves.values() if v is not None] if curves else []

        if not image_list:
            raise ValueError("At least one input image is required")

        # Determine output dimensions
        if size_mode["size_mode"] == "custom":
            out_width = size_mode["width"]
            out_height = size_mode["height"]
        else:
            out_height, out_width = image_list[0].shape[1:3]

        batch_size = image_list[0].shape[0]

        # Prepare batches
        image_batches = []
        for batch_idx in range(batch_size):
            batch_images = [img_tensor[batch_idx].cpu().numpy().astype(np.float32) for img_tensor in image_list]
            image_batches.append(batch_images)

        all_batch_outputs = _render_shader_batch(
            fragment_shader,
            out_width,
            out_height,
            image_batches,
            float_list,
            int_list,
            bool_list,
            curve_luts,
        )

        # Collect outputs into tensors
        all_outputs = [[] for _ in range(MAX_OUTPUTS)]
        for batch_outputs in all_batch_outputs:
            for i, out_img in enumerate(batch_outputs):
                all_outputs[i].append(torch.from_numpy(out_img))

        output_tensors = [torch.stack(all_outputs[i], dim=0) for i in range(MAX_OUTPUTS)]
        return io.NodeOutput(
            *output_tensors,
            ui=cls._build_ui_output(image_list, output_tensors[0]),
        )

    @classmethod
    def _build_ui_output(
        cls, image_list: list[torch.Tensor], output_batch: torch.Tensor
    ) -> dict[str, list]:
        """Build UI output with input and output images for client-side shader execution."""
        input_images_ui = []
        for img in image_list:
            input_images_ui.extend(ui.ImageSaveHelper.save_images(
                img,
                filename_prefix="GLSLShader_input",
                folder_type=io.FolderType.temp,
                cls=None,
                compress_level=1,
            ))

        output_images_ui = ui.ImageSaveHelper.save_images(
            output_batch,
            filename_prefix="GLSLShader_output",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        return {"input_images": input_images_ui, "images": output_images_ui}


class GLSLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GLSLShader]


async def comfy_entrypoint() -> GLSLExtension:
    return GLSLExtension()
