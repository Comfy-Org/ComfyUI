"""Isolated regression of candidate ComfyUI AMD HIP VMM eligibility. No torch imports."""
import ast
import importlib.util
import json
import pathlib
import sys
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_hip_vmm",ROOT/"comfy"/"hip_vmm_capability.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
TREE = ast.parse((ROOT/"main.py").read_text(encoding="utf-8"))
DYNAMIC = next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="dynamic_vram_supported")

class D:
    def __init__(self,index=0,device_type="cuda"):
        self.index=index
        self.type=device_type

def run_dynamic(is_amd=False, is_nvidia=False, version=(7,14), devs=None):
    if devs is None:
        devs=[D()]
    comfy=types.ModuleType("comfy")
    comfy.__path__=[]
    comfy.model_management=types.SimpleNamespace(
        is_nvidia=lambda:is_nvidia,
        is_amd=lambda:is_amd,
        rocm_version=version,
        get_all_torch_devices=lambda:devs)
    with mock.patch.dict(sys.modules,{"comfy":comfy,"comfy.hip_vmm_capability":helper}):
        ns={"comfy":comfy}
        exec(compile(ast.Module(body=[DYNAMIC],type_ignores=[]),"isolated_main.py","exec"),ns)
        return ns["dynamic_vram_supported"]()

class HIPVMMTests(unittest.TestCase):
    def test_attribute_is_89(self):self.assertEqual(helper.HIP_ATTR_VMM,89)
    def test_one_supported(self):self.assertTrue(helper.hip_vmm_supported([D()],query=lambda i:(0,1)))
    def test_two_supported(self):
        calls=[]
        self.assertTrue(helper.hip_vmm_supported([D(0),D(1)],query=lambda i:(calls.append(i) or (0,1))))
        self.assertEqual(calls,[0,1])
    def test_unsupported_zero(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D()],query=lambda i:(0,0)))
    def test_nonzero_status(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D()],query=lambda i:(100,1)))
    def test_ambiguous_support_two(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D()],query=lambda i:(0,2)))
    def test_negative_support(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D()],query=lambda i:(0,-1)))
    def test_partial_multidevice(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D(0),D(1)],query=lambda i:(0,1) if i==0 else (0,0)))
    def test_query_error(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D()],query=lambda i:1/0))
    def test_empty_devices(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([],query=lambda i:(0,1)))
    def test_no_device_iterator(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported(None,query=lambda i:(0,1)))
    def test_enumeration_runtime_failure_fails_closed(self):
        class Fail:
            def __iter__(self):
                raise RuntimeError("GPU enumeration unavailable")
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported(Fail(),query=lambda i:(0,1)))
    def test_bad_index_none(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D(None)],query=lambda i:(0,1)))
    def test_bad_index_negative(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D(-1)],query=lambda i:(0,1)))
    def test_bad_index_bool(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D(True)],query=lambda i:(0,1)))
    def test_bad_device_type(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(helper.hip_vmm_supported([D(device_type="xpu")],query=lambda i:(0,1)))
    def test_duplicate_device_dedup(self):
        calls=[]
        self.assertTrue(helper.hip_vmm_supported([D(0),D(0)],query=lambda i:(calls.append(i) or (0,1))))
        self.assertEqual(calls,[0])
    def test_missing_loaded_runtime_fail_closed(self):
        with mock.patch.object(helper,"_loaded_hip_runtime",return_value=None):
            with self.assertLogs(level="WARNING"):
                self.assertFalse(helper.hip_vmm_supported([D()]))
    def test_missing_export_fail_closed(self):
        with mock.patch.object(helper,"_loaded_hip_runtime",return_value=object()):
            with self.assertLogs(level="WARNING"):
                self.assertFalse(helper.hip_vmm_supported([D()]))
    def test_native_ctypes_abi_and_semantics(self):
        class FakeFunction:
            def __call__(self,pointer,attribute,device):
                self.args_seen=(attribute,device)
                pointer._obj.value=1
                return 0
        func=FakeFunction()
        lib=types.SimpleNamespace(hipDeviceGetAttribute=func)
        with mock.patch.object(helper,"_loaded_hip_runtime",return_value=lib):
            self.assertTrue(helper.hip_vmm_supported([D(2)]))
        self.assertEqual(func.args_seen,(89,2))
        self.assertEqual(func.restype,helper.ctypes.c_int)
        self.assertEqual(len(func.argtypes),3)
    def test_linux_no_loaded_library_returns_none(self):
        fakeos=types.SimpleNamespace(name="posix",RTLD_NOLOAD=4,RTLD_NOW=2,
                                     path=types.SimpleNamespace(basename=lambda p:p.rsplit("/",1)[-1]))
        with mock.patch.object(helper,"os",fakeos),mock.patch.object(helper,"sys",types.SimpleNamespace(platform="linux")):
            with mock.patch("builtins.open",side_effect=OSError("no maps")):
                with mock.patch.object(helper.ctypes,"CDLL",side_effect=OSError("unavailable")) as loaded:
                    self.assertIsNone(helper._loaded_hip_runtime())
                    self.assertTrue(loaded.call_count>=1)
    def test_linux_loaded_library_uses_noload(self):
        fakeos=types.SimpleNamespace(name="posix",RTLD_NOLOAD=4,RTLD_NOW=2,
                                     path=types.SimpleNamespace(basename=lambda p:p.rsplit("/",1)[-1]))
        f=mock.mock_open(read_data="7f0000-7f1000 r-xp 00000000 00:18 123 /opt/rocm/lib/libamdhip64.so.7\n")
        with mock.patch.object(helper,"os",fakeos),mock.patch.object(helper,"sys",types.SimpleNamespace(platform="linux")):
            with mock.patch("builtins.open",f):
                with mock.patch.object(helper.ctypes,"CDLL",return_value="LOADED_EXISTING") as loaded:
                    self.assertEqual(helper._loaded_hip_runtime(),"LOADED_EXISTING")
                    loaded.assert_called_once_with("/opt/rocm/lib/libamdhip64.so.7",mode=6)
    def test_pinned_main_structure(self):
        self.assertIn("from comfy.hip_vmm_capability import hip_vmm_supported",
                      (ROOT/"main.py").read_text(encoding="utf-8"))
    def test_nvidia_path_intact(self):
        with mock.patch.object(helper,"hip_vmm_supported",side_effect=AssertionError("AMD gate called")):
            self.assertTrue(run_dynamic(is_nvidia=True))
    def test_non_amd_and_non_nvidia(self):
        self.assertFalse(run_dynamic())
    def test_amd_version_below_gate_no_query(self):
        with mock.patch.object(helper,"hip_vmm_supported",side_effect=AssertionError("should not call")):
            self.assertFalse(run_dynamic(is_amd=True,version=(7,13)))
    def test_amd_version_at_gate_supported(self):
        with mock.patch.object(helper,"hip_vmm_supported",return_value=True) as q:
            self.assertTrue(run_dynamic(is_amd=True,version=(7,14)))
            q.assert_called_once()
    def test_amd_above_gate_unsupported(self):
        with mock.patch.object(helper,"hip_vmm_supported",return_value=False) as q:
            self.assertFalse(run_dynamic(is_amd=True,version=(7,15)))
            q.assert_called_once()
    def test_amd_multidevice_forwarded(self):
        devs=[D(0),D(1)]
        with mock.patch.object(helper,"hip_vmm_supported",return_value=True) as q:
            self.assertTrue(run_dynamic(is_amd=True,version=(7,15),devs=devs))
            q.assert_called_once_with(devs)
    def test_explicit_override_still_present(self):
        text=(ROOT/"main.py").read_text(encoding="utf-8")
        self.assertIn("if args.enable_dynamic_vram or (enables_dynamic_vram() and dynamic_vram_supported()):",text)
        self.assertIn("Explicit --enable-dynamic-vram bypasses AMD HIP VMM",text)
    def test_modelpatcher_promotion_only_existing_path(self):
        text=(ROOT/"main.py").read_text(encoding="utf-8")
        self.assertEqual(text.count("comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic"),1)
        self.assertEqual(text.count("comfy.memory_management.aimdo_enabled = True"),1)

if __name__=="__main__":unittest.main(verbosity=2)
