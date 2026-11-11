from comfy_api.latest._io import ResolutionPreview
from comfy_extras.nodes_resolution import ResolutionSelector


def test_resolution_preview_defaults_optional_and_socketless():
    preview = ResolutionPreview.Input("preview")

    assert preview.optional is True
    assert preview.socketless is True


def test_resolution_preview_serializes_default_widget_names():
    serialized = ResolutionPreview.Input("preview").as_dict()

    assert serialized["socketless"] is True
    assert serialized["ratio_widget"] == "aspect_ratio"
    assert serialized["megapixels_widget"] == "megapixels"
    assert serialized["multiple_widget"] == "multiple"


def test_resolution_preview_serializes_custom_widget_names():
    serialized = ResolutionPreview.Input(
        "preview",
        ratio_widget="ratio",
        megapixels_widget="mp",
        multiple_widget="resolution_steps",
    ).as_dict()

    assert serialized["ratio_widget"] == "ratio"
    assert serialized["megapixels_widget"] == "mp"
    assert serialized["multiple_widget"] == "resolution_steps"


def test_resolution_preview_carries_no_default_value():
    serialized = ResolutionPreview.Input("preview").as_dict()

    assert "default" not in serialized


def test_resolution_selector_schema_exposes_optional_preview():
    schema = ResolutionSelector.define_schema()
    preview = next(i for i in schema.inputs if i.id == "preview")

    assert preview.optional is True
    assert str(preview.get_io_type()) == "RESOLUTION_PREVIEW"


def test_resolution_selector_executes_without_preview():
    """The frontend never sends the preview value in the API prompt."""
    output = ResolutionSelector.execute("16:9 (Widescreen)", 1.0, 8)

    assert output.result == (1368, 768)


def test_resolution_selector_ignores_preview_value():
    baseline = ResolutionSelector.execute("3:4 (Portrait Standard)", 2.0, 32)
    with_preview = ResolutionSelector.execute(
        "3:4 (Portrait Standard)", 2.0, 32, preview={"stale": True}
    )

    assert baseline.result == with_preview.result == (1248, 1664)
