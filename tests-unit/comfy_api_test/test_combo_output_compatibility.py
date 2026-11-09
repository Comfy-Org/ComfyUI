import asyncio

import pytest

import execution
import nodes
from comfy_api.index import io


def selector(options):
    class Selector(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id='ComboSelectorFixture',
                outputs=[io.Combo.Output('value', options=options)],
            )

        @classmethod
        def execute(cls):
            return io.NodeOutput((options or ['normal'])[0])

    return Selector


@pytest.mark.parametrize('target,accepted', [
    (['normal', 'karras', 'ays'], True),
    (['unrelated'], False),
    ('COMBO', True),
])
def test_declared_combo_output_validates_against_legacy_input(monkeypatch, target, accepted):
    source = selector(['normal', 'karras'])

    class LegacyReceiver:
        RETURN_TYPES = ()
        OUTPUT_NODE = True
        FUNCTION = 'run'

        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'value': (target,)}}

        def run(self, value):
            return {'ui': {'text': [value]}}

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, 'ComboSelectorFixture', source)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, 'LegacyComboReceiverFixture', LegacyReceiver)
    prompt = {
        '1': {'class_type': 'ComboSelectorFixture', 'inputs': {}},
        '2': {'class_type': 'LegacyComboReceiverFixture', 'inputs': {'value': ['1', 0]}},
    }
    valid, _, outputs, errors = asyncio.run(execution.validate_prompt('combo-fixture', prompt, None))
    assert valid is accepted
    if accepted:
        assert outputs == ['2']
        assert errors == {}
    else:
        assert any(error['type'] == 'return_type_mismatch' for error in errors['2']['errors'])


@pytest.mark.parametrize('options,wire_type', [
    (['normal', 'karras'], ['normal', 'karras']),
    ([], 'COMBO'),
    (None, 'COMBO'),
])
def test_combo_projection_keeps_execution_catalog_and_other_outputs_consistent(options, wire_type):
    class MixedOutputs(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id='MixedComboFixture',
                outputs=[
                    io.Combo.Output('choice', options=options, tooltip='Select a mode', is_output_list=True),
                    io.String.Output('text'),
                    io.MatchType.Output(io.MatchType.Template('T', io.String)),
                ],
            )

        @classmethod
        def execute(cls):
            return io.NodeOutput(['normal'], 'text', 'matched')

    info = MixedOutputs.GET_NODE_INFO_V1()
    assert info['output'] == [wire_type, 'STRING', io.MatchType.io_type]
    assert MixedOutputs.RETURN_TYPES == info['output']
    assert info['output_name'] == ['choice', 'text', 'MATCHTYPE']
    assert info['output_is_list'] == [True, False, False]
    assert info['output_tooltips'] == ['Select a mode', None, None]
    assert info['output_matchtypes'] == [None, None, 'T']
