from email.message import Message
from io import BytesIO
import urllib.error
import urllib.request
import urllib.response

import pytest

from comfy_api.latest import _cloud_media
from comfy_api.latest._civitai import InProcessCivitai


@pytest.fixture
def transport(monkeypatch):
    class ScriptedTransport(urllib.request.BaseHandler):
        handler_order = 100

        def __init__(self):
            self.requests = []
            self.responses = {}

        def https_open(self, request):
            self.requests.append(request)
            status, headers, body = self.responses[request.full_url]
            message = Message()
            for name, value in headers.items():
                message[name] = value
            response = urllib.response.addinfourl(
                BytesIO(body), message, request.full_url, status)
            response.msg = 'scripted response'
            return response

        http_open = https_open

    scripted = ScriptedTransport()
    build_opener = urllib.request.build_opener
    legacy = build_opener(scripted)

    def configured_opener(*handlers):
        return build_opener(*handlers, scripted)

    monkeypatch.setattr(urllib.request, 'build_opener', configured_opener)
    monkeypatch.setattr(urllib.request, 'urlopen', legacy.open)
    return scripted


@pytest.mark.parametrize('method,origin,target,status', [
    ('GET', 'https://api.lumalabs.ai/generations/id',
     'https://untrusted.example/result', 302),
    ('POST', 'https://token.sensenova.cn/v1/chat/completions',
     'http://untrusted.example/result', 303),
    ('GET', 'https://api.lumalabs.ai/generations/id',
     'http://169.254.169.254/metadata', 307),
    ('GET', 'https://api.lumalabs.ai/generations/id',
     'https://api.lumalabs.ai/redirected', 308),
])
def test_provider_json_refuses_redirect_without_sending_credentials_to_target(
    transport, method, origin, target, status,
):
    transport.responses[origin] = (status, {'Location': target}, b'')
    transport.responses[target] = (200, {'Content-Type': 'application/json'}, b'{}')
    credential = 'synthetic-provider-credential'
    with pytest.raises(RuntimeError, match=f'HTTP {status}') as failure:
        _cloud_media._request_json(
            method, origin, headers={'Authorization': f'Bearer {credential}'},
            payload={'prompt': 'fixture'} if method == 'POST' else None)
    assert [request.full_url for request in transport.requests] == [origin]
    assert transport.requests[0].get_header('Authorization') == f'Bearer {credential}'
    assert credential not in str(failure.value)
    assert target not in str(failure.value)


@pytest.mark.parametrize('target', [
    'https://untrusted.example/api/v1/models/1',
    'http://127.0.0.1/private',
    'http://169.254.169.254/metadata',
    'https://civitai.com/api/v1/models/2',
])
def test_civitai_refuses_redirect_before_the_target_is_opened(transport, target):
    origin = 'https://civitai.com/api/v1/models/1'
    transport.responses[origin] = (302, {'Location': target}, b'')
    transport.responses[target] = (200, {'Content-Type': 'application/json'}, b'{}')
    with pytest.raises(urllib.error.HTTPError) as failure:
        InProcessCivitai._fetch_json('/api/v1/models/1')
    assert failure.value.code == 302
    assert [request.full_url for request in transport.requests] == [origin]


def test_provider_json_keeps_direct_success_and_response_bounds(transport):
    origin = 'https://api.lumalabs.ai/generations/id'
    transport.responses[origin] = (
        200, {'Content-Type': 'application/json'}, b'{"state":"completed"}')
    assert _cloud_media._request_json('GET', origin) == {'state': 'completed'}
    with pytest.raises(ValueError, match='byte limit'):
        _cloud_media._request_json('GET', origin, maximum=3)


def test_civitai_keeps_direct_success_and_size_validation(transport):
    origin = 'https://civitai.com/api/v1/models/1'
    transport.responses[origin] = (
        200, {'Content-Type': 'application/json'}, b'{"id":1}')
    assert InProcessCivitai._fetch_json('/api/v1/models/1') == {'id': 1}
    transport.responses[origin] = (
        200, {'Content-Type': 'application/json',
              'Content-Length': str(InProcessCivitai._MAX_RESPONSE_BYTES + 1)}, b'{}')
    with pytest.raises(RuntimeError, match='size limit'):
        InProcessCivitai._fetch_json('/api/v1/models/1')
