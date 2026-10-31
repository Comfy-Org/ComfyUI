from unittest.mock import Mock

import comfy.hooks
from comfy_extras.nodes_hooks import SetClipHooks


def test_set_clip_hooks_disables_dynamic_vram_for_scheduled_hooks():
    clip = Mock()
    hooked_clip = Mock()
    clip.clone.return_value = hooked_clip
    hooks = comfy.hooks.HookGroup()

    (result,) = SetClipHooks().apply_hooks(
        clip,
        schedule_clip=True,
        apply_to_conds=True,
        hooks=hooks,
    )

    clip.clone.assert_called_once_with(disable_dynamic=True)
    assert result is hooked_clip
    assert result.apply_hooks_to_conds is hooks
    assert result.use_clip_schedule is True
    assert result.patcher.forced_hooks is not hooks
    result.patcher.register_all_hook_patches.assert_called_once()
