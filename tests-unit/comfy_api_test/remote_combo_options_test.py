import pytest

from comfy_api.latest._io import (
    Combo,
    RemoteComboOptions,
    RemoteItemSchema,
    RemoteOptions,
)


def _schema(**overrides):
    defaults = dict(value_field="id", label_field="name")
    return RemoteItemSchema(**{**defaults, **overrides})


def _combo(**overrides):
    defaults = dict(route="/proxy/foo", item_schema=_schema())
    return RemoteComboOptions(**{**defaults, **overrides})


def test_item_schema_defaults_accepted():
    d = _schema().as_dict()
    assert d == {"value_field": "id", "label_field": "name", "preview_type": "image"}


def test_item_schema_full_config_accepted():
    d = _schema(
        preview_url_field="preview",
        preview_type="audio",
        description_field="desc",
        search_fields=["first", "last", "profile.email"],
    ).as_dict()
    assert d["preview_type"] == "audio"
    assert d["search_fields"] == ["first", "last", "profile.email"]


@pytest.mark.parametrize(
    "bad_fields",
    [
        ["{first} {last}"],
        ["name", "{age}"],
        ["leading{"],
        ["trailing}"],
    ],
)
def test_item_schema_rejects_template_strings_in_search_fields(bad_fields):
    with pytest.raises(ValueError, match="search_fields"):
        _schema(search_fields=bad_fields)


@pytest.mark.parametrize("bad_preview_type", ["middle", "IMAGE", "", "gif"])
def test_item_schema_rejects_unknown_preview_type(bad_preview_type):
    with pytest.raises(ValueError, match="preview_type"):
        _schema(preview_type=bad_preview_type)


def test_combo_options_minimal_accepted():
    d = _combo().as_dict()
    assert d["route"] == "/proxy/foo"
    assert d["refresh_button"] is True
    assert "item_schema" in d


@pytest.mark.parametrize(
    "route",
    [
        "/proxy/foo",
        "/voices",
    ],
)
def test_combo_options_accepts_valid_routes(route):
    _combo(route=route)


@pytest.mark.parametrize(
    "route",
    [
        "",
        "api.example.com/voices",
        "voices",
        "ftp-no-scheme",
        "http://localhost:9000/voices",
        "https://api.example.com/v1/voices",
    ],
)
def test_combo_options_rejects_non_relative_routes(route):
    with pytest.raises(ValueError, match="'route'"):
        _combo(route=route)


@pytest.mark.parametrize("bad_auto_select", ["middle", "FIRST", "", "firstlast"])
def test_combo_options_rejects_unknown_auto_select(bad_auto_select):
    with pytest.raises(ValueError, match="auto_select"):
        _combo(auto_select=bad_auto_select)


@pytest.mark.parametrize("bad_refresh", [1, 127])
def test_combo_options_refresh_in_forbidden_range_rejected(bad_refresh):
    with pytest.raises(ValueError, match="refresh"):
        _combo(refresh=bad_refresh)


@pytest.mark.parametrize("ok_refresh", [0, -1, 128])
def test_combo_options_refresh_valid_values_accepted(ok_refresh):
    _combo(refresh=ok_refresh)


def test_combo_options_timeout_negative_rejected():
    with pytest.raises(ValueError, match="timeout"):
        _combo(timeout=-1)


def test_combo_options_max_retries_negative_rejected():
    with pytest.raises(ValueError, match="max_retries"):
        _combo(max_retries=-1)


def test_combo_options_as_dict_prunes_none_fields():
    d = _combo().as_dict()
    for pruned in ("response_key", "refresh", "timeout", "max_retries", "auto_select"):
        assert pruned not in d


def test_combo_input_accepts_remote_combo_alone():
    Combo.Input("voice", remote_combo=_combo())


def test_combo_input_rejects_remote_plus_remote_combo():
    with pytest.raises(ValueError, match="remote.*remote_combo"):
        Combo.Input(
            "voice",
            remote=RemoteOptions(route="/r", refresh_button=True),
            remote_combo=_combo(),
        )


def test_combo_input_rejects_options_plus_remote_combo():
    with pytest.raises(ValueError, match="options.*remote_combo"):
        Combo.Input("voice", options=["a", "b"], remote_combo=_combo())
