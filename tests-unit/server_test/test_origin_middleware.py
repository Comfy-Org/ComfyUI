import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from middleware.origin_middleware import create_origin_only_middleware


pytestmark = pytest.mark.asyncio


async def ok_handler(request):
    return web.Response(status=200)


async def test_allows_cross_site_top_level_get_navigation():
    request = make_mocked_request(
        'GET',
        '/',
        headers={
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-User': '?1',
        },
    )

    response = await create_origin_only_middleware()(request, ok_handler)

    assert response.status == 200


async def test_blocks_cross_site_top_level_navigation_to_other_routes():
    request = make_mocked_request(
        'GET',
        '/system_stats',
        headers={
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Dest': 'document',
        },
    )

    response = await create_origin_only_middleware()(request, ok_handler)

    assert response.status == 403


@pytest.mark.parametrize(
    ('method', 'headers'),
    [
        (
            'POST',
            {
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Dest': 'document',
            },
        ),
        (
            'GET',
            {
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Dest': 'empty',
            },
        ),
        (
            'GET',
            {
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-Mode': 'websocket',
                'Sec-Fetch-Dest': 'empty',
            },
        ),
        (
            'GET',
            {
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Dest': 'iframe',
            },
        ),
        ('GET', {'Sec-Fetch-Site': 'cross-site'}),
    ],
    ids=['form-post', 'fetch', 'websocket', 'iframe', 'missing-context'],
)
async def test_blocks_other_cross_site_requests(method, headers):
    request = make_mocked_request(method, '/', headers=headers)

    response = await create_origin_only_middleware()(request, ok_handler)

    assert response.status == 403


async def test_preserves_loopback_origin_check_without_fetch_metadata():
    request = make_mocked_request(
        'POST',
        '/prompt',
        headers={
            'Host': '127.0.0.1:8188',
            'Origin': 'https://evil.example',
        },
    )

    response = await create_origin_only_middleware()(request, ok_handler)

    assert response.status == 403


@pytest.mark.parametrize('site', ['same-origin', 'same-site', 'none', None])
async def test_preserves_allowed_request_sources(site):
    headers = {'Sec-Fetch-Site': site} if site is not None else {}
    request = make_mocked_request('GET', '/', headers=headers)

    response = await create_origin_only_middleware()(request, ok_handler)

    assert response.status == 200
