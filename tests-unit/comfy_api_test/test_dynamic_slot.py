"""Unit tests for the redesigned ``DynamicSlot`` with type-keyed options."""

import pytest

from comfy_api.latest import _io as io


def _opt(when, ids=None):
    """Build an Option whose inputs are placeholder String widgets named after ids."""
    ids = ids or []
    inputs = [io.String.Input(name) for name in ids]
    return io.DynamicSlot.Option(when=when, inputs=inputs)


# ---------------------------------------------------------------------------
# Option.when normalization
# ---------------------------------------------------------------------------

def test_option_when_none():
    o = _opt(None, ["a"])
    assert o._when_types is None
    assert o._when_set is None
    assert o.as_dict()["when"] is None


def test_option_when_single_type():
    o = _opt(io.Image)
    assert o._when_types == ("IMAGE",)
    assert o._when_set == frozenset({"IMAGE"})
    assert o.as_dict()["when"] == ["IMAGE"]


def test_option_when_anytype():
    o = _opt(io.AnyType)
    assert o._when_types == ("*",)
    assert o.as_dict()["when"] == ["*"]


def test_option_when_list_preserves_order():
    """Declaration order is preserved in both the tuple and the serialized form."""
    o = _opt([io.Mask, io.Image])
    assert o._when_types == ("MASK", "IMAGE")
    assert o.as_dict()["when"] == ["MASK", "IMAGE"]


def test_option_when_list_dedups_within_option():
    o = _opt([io.Image, io.Image, io.Mask])
    assert o._when_types == ("IMAGE", "MASK")


def test_option_when_multitype_input():
    mt = io.MultiType.Input("x", types=[io.Image, io.Latent])
    o = _opt(mt)
    assert o._when_types == ("IMAGE", "LATENT")


def test_option_when_empty_list_rejected():
    with pytest.raises(ValueError, match="when=\\[\\]"):
        io.DynamicSlot.Option(when=[], inputs=[])


def test_option_when_garbage_rejected():
    with pytest.raises(ValueError, match="when must be"):
        io.DynamicSlot.Option(when="IMAGE", inputs=[])


def test_option_when_list_with_non_comfytype_rejected():
    with pytest.raises(ValueError, match="list entries"):
        io.DynamicSlot.Option(when=[io.Image, "MASK"], inputs=[])


def test_option_when_list_with_anytype_rejected():
    """AnyType must stand alone — it represents the unresolvable-wildcard
    state, not a concrete type that can share a branch with concrete types."""
    with pytest.raises(ValueError, match="AnyType cannot be grouped"):
        io.DynamicSlot.Option(when=[io.Image, io.AnyType], inputs=[])


def test_option_when_multitype_with_anytype_rejected():
    mt = io.MultiType.Input("x", types=[io.Image, io.AnyType])
    with pytest.raises(ValueError, match="AnyType cannot be grouped"):
        io.DynamicSlot.Option(when=mt, inputs=[])


# ---------------------------------------------------------------------------
# DynamicSlot.Input construction and serialization
# ---------------------------------------------------------------------------

def test_input_requires_at_least_one_option():
    with pytest.raises(ValueError, match="at least one Option"):
        io.DynamicSlot.Input("x", options=[])


def test_input_requires_non_none_option():
    with pytest.raises(ValueError, match="non-None `when`"):
        io.DynamicSlot.Input("x", options=[_opt(None, ["a"])])


def test_input_auto_derives_slot_type():
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["a"]),
        _opt(io.Mask, ["b"]),
        _opt(None, ["c"]),
    ])
    # Declared order preserved across non-None options; None contributes nothing.
    assert inp._slot_io_type == "IMAGE,MASK"
    d = inp.as_dict()
    assert d["slotType"] == "IMAGE,MASK"
    assert len(d["options"]) == 3


def test_input_includes_anytype_in_slot_type():
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["a"]),
        _opt(io.AnyType, ["b"]),
    ])
    assert inp._slot_io_type == "IMAGE,*"


def test_input_rejects_duplicate_type_across_options():
    with pytest.raises(ValueError, match="appears in more than one"):
        io.DynamicSlot.Input("x", options=[
            _opt(io.Image, ["a"]),
            _opt([io.Image, io.Mask], ["b"]),
        ])


def test_input_rejects_duplicate_anytype_across_options():
    with pytest.raises(ValueError, match="appears in more than one"):
        io.DynamicSlot.Input("x", options=[
            _opt(io.AnyType, ["a"]),
            _opt(io.AnyType, ["b"]),
        ])


def test_input_rejects_duplicate_when_none():
    with pytest.raises(ValueError, match="only one Option may declare when=None"):
        io.DynamicSlot.Input("x", options=[
            _opt(io.Image, ["a"]),
            _opt(None, ["b"]),
            _opt(None, ["c"]),
        ])


def test_input_rejects_non_option_entry():
    with pytest.raises(ValueError, match="must be DynamicSlot.Option instances"):
        io.DynamicSlot.Input("x", options=[_opt(io.Image, ["a"]), "not an option"])


def test_input_defaults_to_optional_and_always_force_input():
    """The slot is always rendered as a connector, never as a widget, even
    when slotType includes widget-capable types like INT/STRING."""
    inp = io.DynamicSlot.Input("x", options=[_opt(io.Int, ["n"])])
    d = inp.as_dict()
    assert d["forceInput"] is True
    # default optional=True → slot lives in optional bucket via DynamicInput
    assert inp.optional is True


def test_input_required_slot_allowed_without_when_none():
    inp = io.DynamicSlot.Input("x", optional=False, options=[_opt(io.Image, ["a"])])
    assert inp.optional is False


def test_input_required_slot_rejects_when_none_option():
    with pytest.raises(ValueError, match="optional=False forbids when=None"):
        io.DynamicSlot.Input(
            "x",
            optional=False,
            options=[_opt(io.Image, ["a"]), _opt(None, ["b"])],
        )


def test_input_get_all_prepends_self_and_dedups_children():
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["shared", "image_only"]),
        _opt(io.Mask, ["shared", "mask_only"]),
    ])
    items = inp.get_all()
    # Convention shared with Autogrow / DynamicCombo: parent first, then children.
    assert items[0] is inp
    assert [i.id for i in items[1:]] == ["shared", "image_only", "mask_only"]


# ---------------------------------------------------------------------------
# Option selection
# ---------------------------------------------------------------------------

def _select(options, live_input_types, has_link, finalized_id="x"):
    """Convenience wrapper that runs the dispatch through the dict form (post-as_dict)."""
    serialized = [o.as_dict() for o in options]
    return io.DynamicSlot._select_option(
        serialized, live_input_types, finalized_id, has_link
    )


def test_select_unconnected_picks_none_option():
    options = [_opt(io.Image, ["img_widgets"]), _opt(None, ["empty_widgets"])]
    sel = _select(options, {}, has_link=False)
    assert sel is not None
    assert sel["when"] is None


def test_select_unconnected_with_no_none_option_returns_none():
    options = [_opt(io.Image, ["x"])]
    assert _select(options, {}, has_link=False) is None


def test_select_concrete_type_match():
    options = [
        _opt(io.Image, ["a"]),
        _opt(io.Mask, ["b"]),
        _opt(io.AnyType, ["c"]),
    ]
    sel = _select(options, {"x": "MASK"}, has_link=True)
    assert sel["when"] == ["MASK"]


def test_select_anytype_matches_wildcard_resolved():
    options = [_opt(io.Image, ["a"]), _opt(io.AnyType, ["c"])]
    sel = _select(options, {"x": "*"}, has_link=True)
    assert sel["when"] == ["*"]


def test_select_anytype_does_not_match_concrete():
    options = [_opt(io.AnyType, ["c"])]
    # MASK isn't in any option's set; AnyType only matches "*". No expansion.
    assert _select(options, {"x": "MASK"}, has_link=True) is None


def test_select_anytype_branch_does_not_swallow_unenumerated_concrete():
    """Regression: a slot exposing IMAGE + AnyType must reject LATENT upstream
    instead of expanding the AnyType branch. validate_inputs relies on this
    to compute link validity (slotType="IMAGE,*" alone would over-accept)."""
    options = [_opt(io.Image, ["image_widget"]), _opt(io.AnyType, ["any_widget"])]
    assert _select(options, {"x": "LATENT"}, has_link=True) is None
    # Sanity: IMAGE still matches the IMAGE branch and "*" still matches AnyType.
    assert _select(options, {"x": "IMAGE"}, has_link=True)["when"] == ["IMAGE"]
    assert _select(options, {"x": "*"}, has_link=True)["when"] == ["*"]


def test_select_first_match_wins_on_union_upstream():
    """Ordering only matters when upstream declares a multi-type union; with
    per-option type uniqueness, single concrete types can never match two
    options."""
    options = [
        _opt([io.Image, io.Mask], ["image_or_mask"]),
        _opt(io.Latent, ["latent_only"]),
    ]
    # Upstream union "IMAGE,LATENT" intersects both options; first option wins.
    sel = _select(options, {"x": "IMAGE,LATENT"}, has_link=True)
    first_input_id = next(iter(sel["inputs"]["required"].keys()))
    assert first_input_id == "image_or_mask"


def test_select_multitype_upstream_intersects_option_set():
    """When upstream declares MultiType like 'IMAGE,MASK', any option that
    intersects with that set matches (first wins)."""
    options = [
        _opt(io.Latent, ["latent_only"]),
        _opt(io.Mask, ["mask_only"]),
    ]
    sel = _select(options, {"x": "IMAGE,MASK"}, has_link=True)
    assert sel["when"] == ["MASK"]


def test_select_missing_resolved_falls_through_to_anytype():
    """If live_input_types lacks an entry for this slot but a link exists,
    we treat it as '*' (resolver default for unresolvable links)."""
    options = [_opt(io.Image, ["a"]), _opt(io.AnyType, ["c"])]
    sel = _select(options, {}, has_link=True)
    assert sel["when"] == ["*"]


# ---------------------------------------------------------------------------
# End-to-end expansion via _expand_schema_for_dynamic
# ---------------------------------------------------------------------------

def test_expand_unconnected_path():
    """An unconnected slot with a `when=None` option expands that option's children."""
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["image_widget"]),
        _opt(None, ["empty_widget"]),
    ])
    d = inp.as_dict()
    value = (io.DynamicSlot.io_type, d)
    out_dict = {
        "required": {}, "optional": {}, "hidden": {},
        "dynamic_paths": {}, "dynamic_paths_default_value": {},
    }
    io.DynamicSlot._expand_schema_for_dynamic(
        out_dict=out_dict,
        live_inputs={},  # no entry for "x" → unconnected
        value=value,
        input_type="optional",
        curr_prefix=["x"],
        live_input_types=None,
    )
    # The slot itself is always advertised in the caller's bucket.
    assert "x" in out_dict["optional"]
    # Children land in their own buckets (required by default) with
    # parent-prefixed ids.
    assert "x.empty_widget" in out_dict["required"]
    assert "x.image_widget" not in out_dict["required"]


def test_expand_typed_path():
    """A connected slot expands the matching type's children."""
    inp = io.DynamicSlot.Input("x", options=[
        _opt(io.Image, ["image_widget"]),
        _opt(io.Mask, ["mask_widget"]),
    ])
    d = inp.as_dict()
    value = (io.DynamicSlot.io_type, d)
    out_dict = {
        "required": {}, "optional": {}, "hidden": {},
        "dynamic_paths": {}, "dynamic_paths_default_value": {},
    }
    io.DynamicSlot._expand_schema_for_dynamic(
        out_dict=out_dict,
        live_inputs={"x": ["src_node", 0]},  # link present
        value=value,
        input_type="optional",
        curr_prefix=["x"],
        live_input_types={"x": "MASK"},
    )
    assert "x" in out_dict["optional"]
    assert "x.mask_widget" in out_dict["required"]
    assert "x.image_widget" not in out_dict["required"]


def test_expand_unmatched_concrete_still_advertises_slot():
    """Resolved type not in any option → no children, but the slot itself stays."""
    inp = io.DynamicSlot.Input("x", options=[_opt(io.Image, ["image_widget"])])
    d = inp.as_dict()
    value = (io.DynamicSlot.io_type, d)
    out_dict = {
        "required": {}, "optional": {}, "hidden": {},
        "dynamic_paths": {}, "dynamic_paths_default_value": {},
    }
    io.DynamicSlot._expand_schema_for_dynamic(
        out_dict=out_dict,
        live_inputs={"x": ["src_node", 0]},
        value=value,
        input_type="optional",
        curr_prefix=["x"],
        live_input_types={"x": "LATENT"},
    )
    assert "x" in out_dict["optional"]
    assert "x.image_widget" not in out_dict["required"]
