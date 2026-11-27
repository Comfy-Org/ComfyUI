"""GPU regression: python -m unittest discover -s tests-unit/comfy_test -p rocm_mmap_upload_test.py."""
import concurrent.futures
import gc
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

if not (sys.platform == 'linux' and torch.version.hip and torch.cuda.is_available()):
    raise unittest.SkipTest('Linux ROCm GPU required')

import safetensors.torch
from comfy import model_management as mm
from comfy_kitchen.tensor.base import BaseLayoutParams, QuantizedTensor


def anonymous_bytes(path):
    active = False
    total = 0
    for line in Path('/proc/self/smaps').read_text().splitlines():
        parts = line.split()
        if parts and '-' in parts[0] and len(parts) >= 5:
            active = line.rstrip().endswith(str(path))
        elif active and line.startswith('Anonymous:'):
            total += int(parts[1]) * 1024
    return total


class MmapUploadTest(unittest.TestCase):
    def transfer(self, *, transpose=False, stream=None, preallocated=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.safetensors'
            codes = torch.arange(256, dtype=torch.uint8).repeat(32768).reshape(4096, 2048)
            safetensors.torch.save_file({'codes': codes, 'scale': torch.tensor([0.25])}, path)
            tensors = safetensors.torch.load_file(path)
            data = tensors['codes'].t() if transpose else tensors['codes']
            source = QuantizedTensor(data, 'test', BaseLayoutParams(tensors['scale'], torch.float16, tuple(data.shape)))
            before = anonymous_bytes(path)
            destination = torch.empty_like(source, device='cuda:0', dtype=torch.bfloat16) if preallocated else None
            result = mm.cast_to(source, dtype=torch.bfloat16, device=torch.device('cuda:0'),
                                non_blocking=True, stream=stream, r=destination)
            torch.cuda.synchronize()
            self.assertEqual(anonymous_bytes(path), before, 'Upload created private copies of the file mapping')
            self.assertTrue(torch.equal(result._qdata.cpu(), data))
            self.assertTrue(torch.equal(result._params.scale.cpu(), tensors['scale']))
            self.assertEqual(result._params.orig_dtype, torch.bfloat16)
            self.assertEqual(result.dtype, torch.bfloat16)
            self.assertEqual(result._params.orig_shape, tuple(data.shape))
            self.assertEqual(result._layout_cls, source._layout_cls)
            self.assertTrue(torch.equal(data, codes.t() if transpose else codes))
            self.assertEqual(source.dtype, torch.float16)
            if preallocated:
                self.assertIs(result, destination)
            del source, result, destination, data, tensors
            gc.collect()

    def test_mapped_upload(self):
        self.transfer()

    def test_strided_upload_on_caller_stream_into_existing_destination(self):
        self.transfer(transpose=True, stream=torch.cuda.Stream(), preallocated=True)

    def test_concurrent_transfers_have_independent_staging(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.transfer, stream=torch.cuda.Stream()) for _ in range(2)]
            for future in futures:
                future.result()

    def test_cpu_and_anonymous_sources(self):
        source = QuantizedTensor(torch.arange(256, dtype=torch.uint8), 'test', BaseLayoutParams(torch.tensor(0.5), torch.float16, (256,)))
        self.assertIs(mm.cast_to(source, device=torch.device('cpu')), source)
        result = mm.cast_to(source, device=torch.device('cuda:0'), non_blocking=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(result._qdata.cpu(), source._qdata))
        self.assertEqual(result._params.scale.item(), 0.5)

    def test_oversized_weight_warns_and_preserves_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.safetensors'
            safetensors.torch.save_file({'codes': torch.arange(256, dtype=torch.uint8)}, path)
            data = safetensors.torch.load_file(path)['codes']
            source = QuantizedTensor(data, 'test', BaseLayoutParams(torch.tensor(1.0), torch.float16, (256,)))
            with mock.patch.object(mm.comfy.memory_management, 'vram_aligned_size', return_value=513 * 1024**2), self.assertLogs(level='WARNING'):
                result = mm.cast_to(source, device=torch.device('cuda:0'))
            self.assertTrue(torch.equal(result._qdata.cpu(), data))
            del source, result, data
            gc.collect()

    def test_non_rocm_does_not_inspect_mappings(self):
        source = QuantizedTensor(torch.arange(256, dtype=torch.uint8), 'test', BaseLayoutParams(torch.tensor(1.0), torch.float16, (256,)))
        with mock.patch.object(mm, 'is_amd', return_value=False), mock.patch('builtins.open', side_effect=AssertionError('Unexpected mapping inspection')):
            result = mm.cast_to(source, device=torch.device('cuda:0'))
        self.assertTrue(torch.equal(result._qdata.cpu(), source._qdata))


if __name__ == '__main__':
    unittest.main()
