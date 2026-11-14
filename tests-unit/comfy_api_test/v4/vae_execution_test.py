import pytest

from comfy.cli_args import args


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
async def test_a_vae_built_by_a_pack_round_trips_through_comfyuis_vae_nodes(
    run_node, monkeypatch, synchronous, tiled
):
    monkeypatch.setattr(args, "cpu_vae", True)
    monkeypatch.setattr(args, "fp32_vae", True)
    suffix, options = (
        ("_tiled", ", tile_x=8, tile_y=8, overlap=2") if tiled else ("", "")
    )

    executor, _ = await run_node(
        "pixels = torch.linspace(0, 1, 2 * 12 * 16 * 3).reshape(2, 12, 16, 3)\n"
        'vae = {call}.vae.from_state_dict({{"pixel_space_vae": torch.tensor(1.0)}})\n'
        '(latent,) = {call}.node.run("VAEEncode", {{"vae": vae, "pixels": pixels}})\n'
        f'decoded = {{call}}.vae.decode{suffix}(vae, latent["samples"]{options})\n'
        f"samples = {{call}}.vae.encode{suffix}(vae, decoded{options})\n"
        '(restored,) = {call}.node.run("VAEDecode", {{"vae": vae, "samples": {{"samples": samples}}}})\n'
        "torch.testing.assert_close(restored, pixels)",
        synchronous,
    )

    assert executor.success, executor.status_messages
