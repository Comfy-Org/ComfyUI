"""Converted route declarations must be visible to the host registration reader."""
import importlib
import json
import sys
import types

import pytest

from comfy_api.latest import registration
import convert
import verify


@pytest.mark.parametrize('with_project', [True, False])
def test_converted_routes_are_declared_and_mounted(tmp_path, monkeypatch, with_project):
    pack = tmp_path / 'source-pack'
    pack.mkdir()
    (pack / '__init__.py').write_text('NODE_CLASS_MAPPINGS = {}\n')
    if with_project:
        (pack / 'pyproject.toml').write_text(
            '[project]\nname = "preserved-name"\nversion = "1.2.3"\n'
            'dependencies = ["example-dependency>=1"]\n'
            '[tool.comfy]\nPublisherId = "original-publisher"\n'
        )
    describe = {'nodes': {}, 'routes': [['GET', '/fixture/status']], 'web_directory': None}
    report = convert.generate(str(pack), describe, str(tmp_path / 'converted'))
    v2 = tmp_path / 'converted' / 'source-pack' / 'v2'
    declared = registration.declarations(str(v2))
    route = {'method': 'get', 'path': '/fixture/status', 'module': '_v2_routes', 'handler': 'handle'}
    assert route in declared['comfy']['routes']
    assert not verify.missing_generated_routes(describe, declared['comfy']['routes'])
    if with_project:
        assert declared['project']['name'] == 'preserved-name'
        assert declared['project']['version'] == '1.2.3'
        assert declared['project']['dependencies'] == ['example-dependency>=1']
        assert declared['comfy']['PublisherId'] == 'original-publisher'
    assert (v2 / 'pyproject.toml').is_file()

    # Load only the generated route module from the trusted synthetic fixture.
    package = '_route_fixture'
    module = types.ModuleType(package)
    module.__path__ = [str(v2)]
    monkeypatch.setitem(sys.modules, package, module)
    monkeypatch.setitem(sys.modules, package + '._v2_engine', types.ModuleType(package + '._v2_engine'))
    monkeypatch.delitem(sys.modules, package + '._v2_routes', raising=False)
    mounted = []
    class Router:
        def route(self, method, url):
            def register(handler):
                mounted.append((method, url, handler))
                return handler
            return register
    server = types.SimpleNamespace(routes=Router())
    namespace = report['namespace']
    expected = f'GET /ext/{namespace}/fixture/status'
    assert registration.install_routes(str(v2), package, server) == [expected]
    assert [(method, url) for method, url, _ in mounted] == [('GET', f'/ext/{namespace}/fixture/status')]
    assert callable(mounted[0][2])
    monkeypatch.delitem(sys.modules, package + '._v2_routes', raising=False)


def test_verifier_rejects_missing_generated_route_declarations(tmp_path, monkeypatch):
    describe = {'nodes': {}, 'routes': [['GET', '/fixture/status'], ['HEAD', '/fixture/status'],
                                       ['POST', '/variable/{id}']]}
    described = tmp_path / 'describe.json'
    described.write_text(json.dumps(describe))
    out = tmp_path / 'verify.json'
    declarations = tmp_path / 'declarations.json'
    monkeypatch.setattr(verify, '_scratch', lambda _: str(declarations))
    def child(args, timeout):
        declarations.write_text(json.dumps({'valid': True, 'declared_routes': [], 'routes': {}}))
        return 0, ''
    monkeypatch.setattr(verify, '_child', child)
    verify.main(str(tmp_path), str(tmp_path), str(described), str(out))
    result = json.loads(out.read_text())
    assert result['valid'] is False
    assert result['missing_declared_routes'] == ['GET /fixture/status']
    assert result['routes'] == {}
    assert 'declarations missing' in result['error']
