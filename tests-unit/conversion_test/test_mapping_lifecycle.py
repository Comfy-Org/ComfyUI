"""A pack scanner may import generated wrappers before populating its table."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import subprocess
import sys
import textwrap

import convert


def test_early_wrapper_import_keeps_final_original_classes(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '__init__.py').write_text(textwrap.dedent('''
        import importlib
        import os
        NODE_CLASS_MAPPINGS = {}
        for filename in sorted(os.listdir(os.path.dirname(__file__))):
            if filename.endswith('.py') and filename != '__init__.py':
                module = importlib.import_module('.' + filename[:-3], __name__)
                if filename == '_v2_nodes.py':
                    EARLY_CAPTURE = dict(module._V1)
                NODE_CLASS_MAPPINGS.update(getattr(module, 'NODE_CLASS_MAPPINGS', {}))
    '''))
    (source / 'z_nodes.py').write_text(textwrap.dedent('''
        class AddOne:
            FUNCTION = 'run'
            RETURN_TYPES = ('INT',)
            CATEGORY = 'test'
            @classmethod
            def INPUT_TYPES(cls):
                return {'required': {'value': ('INT', {'default': 3})}}
            def run(self, value):
                return (value + 1,)
        NODE_CLASS_MAPPINGS = {'EarlyScannerAddOne': AddOne}
    '''))
    info = {
        'input': {'required': {'value': ['INT', {'default': 3}]}},
        'input_order': {'required': ['value']}, 'is_input_list': False,
        'output': ['INT'], 'output_is_list': [False], 'output_name': ['INT'],
        'name': 'EarlyScannerAddOne', 'display_name': 'EarlyScannerAddOne',
        'description': '', 'category': 'test', 'output_node': False,
        'has_intermediate_output': False, 'search_aliases': [],
    }
    description = {'ok': True, 'nodes': {'EarlyScannerAddOne': {
        'class': 'AddOne', 'kind': 'v1', 'info': info, 'hints': {},
        'function': 'run', 'methods': [], 'module': '/pack.z_nodes',
        'qualname': 'AddOne', 'file': '/pack/z_nodes.py',
        'params': ['self', 'value'],
    }}, 'web_directory': None, 'routes': []}
    output = tmp_path / 'converted'
    convert.generate(str(source), description, str(output))
    # ComfyUI parses its CPU setting at first import. Use a fresh process so
    # other tests importing cli_args cannot change the real loader's startup.
    check = textwrap.dedent('''
        import asyncio
        import sys
        pack_path = sys.argv[1]
        from describe import cpu_comfy
        cpu_comfy(as_comfyui_loads_packs=True)
        from packload import LoadedPack
        pack = LoadedPack(pack_path)
        module = sys.modules[pack.module_name]
        assert module.EARLY_CAPTURE == {}
        assert set(module._V2_ORIGINAL_MAPPINGS) == {'EarlyScannerAddOne'}
        assert module._V2_node_module._V1 == module._V2_ORIGINAL_MAPPINGS
        assert set(pack.nodes) == {'EarlyScannerAddOne'}
        result = asyncio.run(pack.nodes['EarlyScannerAddOne'].execute(value=41))
        assert tuple(result.result) == (42,)
    ''')
    result = subprocess.run([sys.executable, '-B', '-c', check,
                             str(output / 'source' / 'v2')],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
