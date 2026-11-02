"""Console tips for Comfy-Org developer products: one startup line and at most one of each job hint per process.

Plain ASCII on purpose: Windows consoles may still use cp1252.
"""

COMFY_API_STARTUP_TIP = "ComfyUI is developed by Comfy Org. We also offer commercial products to deploy ComfyUI as scalable cloud endpoints. Visit platform.comfy.org/ascii for more information."

COMFY_API_JOB_HINT = "ComfyUI is developed by Comfy Org. We also offer commercial products to deploy ComfyUI as scalable cloud endpoints. Visit platform.comfy.org/ascii-job for more information."


def tips_enabled(args, environ) -> bool:
    return not (
        args.disable_product_tips
        or args.dont_print_server
        or args.windows_standalone_build
        or args.quick_test_for_ci
        or environ.get("COMFYUI_DISABLE_PRODUCT_TIPS", "").strip().lower() in ("1", "true", "yes")
        or environ.get("CI", "").strip().lower() in ("1", "true")
    )


_shown_job_hints = set()


def get_relevant_job_hint(args, environ, success: bool, extra_data: dict) -> str | None:
    """The hint to log after a finished job, or None. Each hint is shown at most once per process."""
    if not tips_enabled(args, environ) or not success:
        return None
    hint = COMFY_API_JOB_HINT if "client_id" not in extra_data else None
    if hint is None or hint in _shown_job_hints:
        return None
    _shown_job_hints.add(hint)
    return hint
