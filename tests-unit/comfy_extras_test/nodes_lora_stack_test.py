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
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_filename_list', lambda _: ['a.safetensors', 'b.safetensors'])
    schema = create_input_dict_v1(node_class.define_schema().inputs)
    values = {
        'loras.0.lora_name': 'a.safetensors', 'loras.0.strength': 0.5,
        'loras.2.lora_name': 'b.safetensors', 'loras.2.strength': -0.25,
    }
    _, _, v3_data = get_finalized_class_inputs(schema, values)
    rows = build_nested_inputs(values, v3_data)['loras']
    assert rows[1] == {'lora_name': None, 'strength': None}

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
def test_skips_empty_positions_and_zero_strength_without_loading(monkeypatch, node_class):
    load = Mock()
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_full_path_or_raise', load)
    original = object()
    result = node_class.execute(original, [
        {'lora_name': None, 'strength': None},
        {'lora_name': 'disabled.safetensors', 'strength': 0},
    ])
    assert result.result == (original,)
    load.assert_not_called()


@pytest.mark.parametrize('node_class', [nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder])
def test_missing_lora_fails_at_the_file_resolver(monkeypatch, node_class):
    resolve = Mock(side_effect=FileNotFoundError('Missing LoRA'))
    load = Mock()
    monkeypatch.setattr(nodes_lora_stack.folder_paths, 'get_full_path_or_raise', resolve)
    monkeypatch.setattr(nodes_lora_stack.comfy.utils, 'load_torch_file', load)
    with pytest.raises(FileNotFoundError, match='Missing LoRA'):
        node_class.execute(object(), [{'lora_name': 'missing.safetensors', 'strength': 1}])
    resolve.assert_called_once_with('loras', 'missing.safetensors')
    load.assert_not_called()


def test_extension_exports_both_loaders():
    extension = asyncio.run(nodes_lora_stack.comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [
        nodes_lora_stack.LoadLoraModel, nodes_lora_stack.LoadLoraTextEncoder,
    ]
