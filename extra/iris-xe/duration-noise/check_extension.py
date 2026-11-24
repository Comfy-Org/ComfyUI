"""Check extension, interrupted/restarted reuse, masking, and decoded MP4 prefixes on CPU."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import comfy.options
comfy.options.enable_args_parsing()
import av
import numpy as np
import torch
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
import ltx_text_continuation as ext


def read_rgb(video):
    with av.open(video.get_stream_source()) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def check():
    calls = []
    def sample(noise, guider, sampler, sigmas, latent):
        x = latent["samples"]
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
            assert x.shape[2] == 4 and torch.all(mask[:, :, :2] == 0) and torch.all(mask[:, :, 2:] == 1)
            assert torch.equal(x[:, :, :2], calls[-1][:, :, -2:])
        x = noise.generate_noise(latent)
        calls.append(x.clone())
        return ({"samples": x},)

    def decode(vae, latent, *args):
        x = latent["samples"]
        times = 1 + (x.shape[2]-1)*8
        frames = torch.rand((times, 64, 64, 3), generator=torch.Generator().manual_seed(len(calls)))
        return (frames,)

    prompt = {"12":{"inputs":{name:[str(i),0] for i,name in enumerate(("noise","guider","sampler","sigmas","vae"))}}}
    prompt.update({str(i):{"class_type":"Test","inputs":{"v":i}} for i in range(5)})
    with tempfile.TemporaryDirectory() as directory, patch.object(ext.folder_paths,"get_output_directory",return_value=directory), patch.object(ext.folder_paths,"get_temp_directory",return_value=directory), patch.object(ext.SamplerCustomAdvanced,"execute",side_effect=sample), patch.object(ext.VAEDecodeTiled,"decode",side_effect=decode):
        def render(length, unit="frames"):
            video, = ext.LTXTextContinuation().render(Noise_RandomNoise(3),None,None,None,None,64,64,length,unit,24,prompt,"12")
            return read_rgb(video)

        previous = None
        for length, expected_calls in [(9,1),(17,2),(25,2),(225,15)]:
            result = render(length)
            assert len(calls)==expected_calls and len(result)==length
            if previous is not None:
                assert np.array_equal(previous,result[:len(previous)])
            previous=result
        seconds = render(9,"seconds")
        assert len(calls)==15 and len(seconds)==216 and np.array_equal(previous[:216],seconds)
        # New prompt must start a new shot rather than return stale disk data.
        prompt["1"]["inputs"]["v"] = "changed"
        render(9)
        assert len(calls)==16


if __name__=="__main__":
    check()
    print("PASS: conditioning mask, new shot on prompt change, reuse across calls and exact decoded MP4 prefixes")
