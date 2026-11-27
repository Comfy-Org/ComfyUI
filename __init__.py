import sys
import os
import importlib
import importlib.abc
import importlib.machinery
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))

# Các sub-package của ComfyUI (utils, app, ...) được import theo kiểu flat,
# ví dụ: `from utils.install_util import ...` thay vì `from ComfyUI.utils...`.
# Trên Colab/Kaggle có thể tồn tại package/module trùng tên (ví dụ PyPI 'utils')
# đã được cache vào sys.modules → gây lỗi "utils is not a package".
# Giải pháp: xóa mọi entry xung đột, thêm _here vào sys.path[0], VÀ nạp
# trực tiếp 'utils' từ đường dẫn tuyệt đối thay vì dùng sys.path search.

_subpkg_names = ('utils', 'app', 'api_server', 'middleware', 'comfy', 'comfy_api',
                 'comfy_execution', 'comfy_extras', 'comfy_config')
for _n in _subpkg_names:
    sys.modules.pop(_n, None)

# Đảm bảo _here luôn ở đầu sys.path (fallback cho các import không được đăng ký bên dưới).
try:
    sys.path.remove(_here)
except ValueError:
    pass
sys.path.insert(0, _here)


class _ComfyUIFlatImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """
    Đảm bảo `import ComfyUI.X` và `import X` luôn trả về CÙNG một module object
    trong sys.modules, chia sẻ toàn bộ global state (ví dụ: folder_paths,
    server.PromptServer.instance, nodes.NODE_CLASS_MAPPINGS, ...).

    Nguyên nhân gốc: khi _here được thêm vào sys.path, Python tạo ra 2 entry riêng
    biệt — sys.modules['folder_paths'] và sys.modules['ComfyUI.folder_paths'] — mỗi
    entry có namespace riêng, khiến việc cập nhật global ở một phía không ảnh hưởng
    phía kia.
    """
    _PREFIX = 'ComfyUI.'

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self._PREFIX):
            return None
        flat_name = fullname[len(self._PREFIX):]
        # Module đã được nạp dưới tên flat → alias ngay
        if flat_name in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self)
        # Module chưa nạp nhưng file tồn tại trong _here → để create_module nạp flat trước
        parts = flat_name.split('.')
        if (os.path.isfile(os.path.join(_here, *parts) + '.py') or
                os.path.isfile(os.path.join(_here, *parts, '__init__.py'))):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        flat_name = spec.name[len(self._PREFIX):]
        if flat_name not in sys.modules:
            # Nạp dưới tên flat, _here đã ở sys.path[0] nên sẽ tìm đúng file
            importlib.import_module(flat_name)
        return sys.modules[flat_name]

    def exec_module(self, module):
        # Module đã được exec khi nạp dưới tên flat; không làm gì thêm.
        pass


# Cài importer ở vị trí 0 để ưu tiên cao nhất, trước PathFinder của Python
sys.meta_path.insert(0, _ComfyUIFlatImporter())


# Nạp trực tiếp 'utils' package từ đường dẫn tuyệt đối, hoàn toàn bỏ qua
# mọi sys.path conflict. Điều này đảm bảo `from utils.install_util import ...`
# luôn trỏ đúng vào ComfyUI/utils/, kể cả khi môi trường có utils.py hoặc
# utils/ khác được cài trước.
def _load_subpkg_as_toplevel(name: str) -> None:
    pkg_dir = os.path.join(_here, name)
    init_file = os.path.join(pkg_dir, '__init__.py')
    if not os.path.isfile(init_file):
        return
    spec = _ilu.spec_from_file_location(name, init_file,
                                         submodule_search_locations=[pkg_dir])
    if spec is None:
        return
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod                          # alias top-level
    sys.modules.setdefault(f'ComfyUI.{name}', mod)  # alias ComfyUI.<name>
    spec.loader.exec_module(mod)

_load_subpkg_as_toplevel('utils')
del _ilu, _load_subpkg_as_toplevel
