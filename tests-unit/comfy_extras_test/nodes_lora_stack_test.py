import asyncio
from unittest.mock import Mock, call

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api.latest._io import build_nested_inputs, create_input_dict_v1, get_finalized_class_inputs
from comfy_extras import nodes_lora_stack


@pytest.mark.parametrize('node_class,target', [
    (nodes_lora_stack.LoadLoraModel, 'model'),
    (nodes_lora_stack.LoadLoraTextEncoder, 'clip'),
])
def test_applies_sparse_rows_in_order_with_metadata(monkeypatch, node_class, target):
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_filename_list', lambda _: ['a.safetensors', 'b.safetensors', 'disabled.safetensors'])
    schema = create_input_dict_v1(node_class.define_schema().inputs)
    values = {
        'loras.0.lora_name': 'a.safetensors', 'loras.0.strength': 0.5, 'loras.0.enabled': True,
        'loras.2.lora_name': 'disabled.safetensors', 'loras.2.strength': 0.75, 'loras.2.enabled': False,
        'loras.3.lora_name': 'b.safetensors', 'loras.3.strength': -0.25, 'loras.3.enabled': True,
    }
    _, _, v3_data = get_finalized_class_inputs(schema, values)
    rows = build_nested_inputs(values, v3_data)['loras']
    assert rows[1] == {'lora_name': None, 'strength': None, 'enabled': None}

    original, first, second = object(), object(), object()
    lora_a, lora_b = object(), object()
    metadata_a, metadata_b = {'name': 'a'}, {'name': 'b'}
    resolve = Mock(side_effect=['/loras/a.safetensors', '/loras/b.safetensors'])
    load = Mock(side_effect=[(lora_a, metadata_a), (lora_b, metadata_b)])
    apply = Mock(side_effect=[(first, None), (second, None)] if target == 'model' else [(None, first), (None, second)])
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_full_path_or_raise', resolve)
    monkeypatch.setattr(nodes_lora_stack.comfy.utils, 'load_torch_file', load)
    monkeypatch.setattr(nodes_lora_stack.comfy.sd, 'load_lora_for_models', apply)

    result = node_class.execute(original, rows)

    assert result.result == (second,)
    assert resolve.call_args_list == [call('loras', 'a.safetensors'), call('loras', 'b.safetensors')]
    assert load.call_args_list == [
        call('/loras/a.safetensors', safe_load=True, return_metadata=True),
        call('/loras/b.safetensors', safe_load=True, return_metadata=True),
    ]
    if target == 'model':
        assert apply.call_args_list == [
            call(original, None, lora_a, 0.5, 0, lora_metadata=metadata_a),
            call(first, None, lora_b, -0.25, 0, lora_metadata=metadata_b),
        ]
    else:
        assert apply.call_args_list == [
            call(None, original, lora_a, 0, 0.5, lora_metadata=metadata_a),
            call(None, first, lora_b, 0, -0.25, lora_metadata=metadata_b),
        ]


@pytest.mark.parametrize('node_class', [nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder])
def test_accepts_twentieth_position_and_rejects_later_indices(monkeypatch, node_class):
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_filename_list', lambda _: ['a.safetensors'])
    schema = create_input_dict_v1(node_class.define_schema().inputs)
    values = {'loras.19.lora_name': 'a.safetensors', 'loras.19.strength': 0.5, 'loras.19.enabled': True}
    _, _, v3_data = get_finalized_class_inputs(schema, values)
    rows = build_nested_inputs(values, v3_data)['loras']
    assert rows == [{'lora_name': None, 'strength': None, 'enabled': None}] * 19 + [{'lora_name': 'a.safetensors', 'strength': 0.5, 'enabled': True}]

    with pytest.raises(ValueError, match='exceeds the index limit of 19'):
        get_finalized_class_inputs(schema, {'loras.20.lora_name': 'a.safetensors', 'loras.20.strength': 0.5, 'loras.20.enabled': True})


@pytest.mark.parametrize('node_class', [nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder])
def test_skips_empty_positions_and_zero_strength_without_loading(monkeypatch, node_class):
    load = Mock()
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_full_path_or_raise', load)
    original = object()
    result = node_class.execute(original, [
        {'lora_name': None, 'strength': None, 'enabled': None},
        {'lora_name': 'zero.safetensors', 'strength': 0, 'enabled': True},
    ])
    assert result.result == (original,)
    load.assert_not_called()


@pytest.mark.parametrize('node_class,target', [
    (nodes_lora_stack.LoadLoraModel, 'model'),
    (nodes_lora_stack.LoadLoraTextEncoder, 'clip'),
])
def test_can_reenable_row_without_changing_file_or_strength(monkeypatch, node_class, target):
    original, patched, lora = object(), object(), object()
    resolve = Mock(return_value='/loras/a.safetensors')
    load = Mock(return_value=(lora, None))
    apply = Mock(return_value=(patched, None) if target == 'model' else (None, patched))
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_full_path_or_raise', resolve)
    monkeypatch.setattr(nodes_lora_stack.comfy.utils, 'load_torch_file', load)
    monkeypatch.setattr(nodes_lora_stack.comfy.sd, 'load_lora_for_models', apply)
    row = {'lora_name': 'a.safetensors', 'strength': -0.75, 'enabled': False}

    assert node_class.execute(original, [row]).result == (original,)
    resolve.assert_not_called()
    load.assert_not_called()
    apply.assert_not_called()
    assert row == {'lora_name': 'a.safetensors', 'strength': -0.75, 'enabled': False}

    row['enabled'] = True
    assert node_class.execute(original, [row]).result == (patched,)
    resolve.assert_called_once_with('loras', 'a.safetensors')
    load.assert_called_once_with('/loras/a.safetensors', safe_load=True, return_metadata=True)
    if target == 'model':
        apply.assert_called_once_with(original, None, lora, -0.75, 0, lora_metadata=None)
    else:
        apply.assert_called_once_with(None, original, lora, 0, -0.75, lora_metadata=None)


@pytest.mark.parametrize('node_class', [nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder])
def test_new_rows_default_to_enabled(node_class):
    schema = node_class.INPUT_TYPES()
    template = schema['required']['loras'][1]['template']['required']
    assert template['enabled'][0] == 'BOOLEAN'
    assert template['enabled'][1]['default'] is True


@pytest.mark.parametrize('node_class', [nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder])
def test_missing_lora_fails_at_the_file_resolver(monkeypatch, node_class):
    resolve = Mock(side_effect=FileNotFoundError('Missing LoRA'))
    load = Mock()
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_full_path_or_raise', resolve)
    monkeypatch.setattr(nodes_lora_stack.comfy.utils, 'load_torch_file', load)
    with pytest.raises(FileNotFoundError, match='Missing LoRA'):
        node_class.execute(object(), [{'lora_name': 'missing.safetensors', 'strength': 1, 'enabled': True}])
    resolve.assert_called_once_with('loras', 'missing.safetensors')
    load.assert_not_called()


def test_extension_exports_both_loaders():
    extension = asyncio.run(nodes_lora_stack.comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [
        nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder,
    ]
