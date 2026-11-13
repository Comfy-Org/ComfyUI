import pygit2
from datetime import datetime
import sys
import os
import shutil
import filecmp
import importlib.util
import tempfile

def pull(repo, remote_name='origin', branch='master'):
    for remote in repo.remotes:
        if remote.name == remote_name:
            remote.fetch()
            remote_master_id = repo.lookup_reference('refs/remotes/origin/%s' % (branch)).target
            merge_result, _ = repo.merge_analysis(remote_master_id)
            # Up to date, do nothing
            if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
                return
            # We can just fastforward
            elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
                repo.checkout_tree(repo.get(remote_master_id))
                try:
                    master_ref = repo.lookup_reference('refs/heads/%s' % (branch))
                    master_ref.set_target(remote_master_id)
                except KeyError:
                    repo.create_branch(branch, repo.get(remote_master_id))
                repo.head.set_target(remote_master_id)
            elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
                repo.merge(remote_master_id)

                if repo.index.conflicts is not None:
                    for conflict in repo.index.conflicts:
                        print('Conflicts found in:', conflict[0].path)  # noqa: T201
                    raise AssertionError('Conflicts, ahhhhh!!')

                user = repo.default_signature
                tree = repo.index.write_tree()
                repo.create_commit('HEAD',
                                    user,
                                    user,
                                    'Merge!',
                                    tree,
                                    [repo.head.target, remote_master_id])
                # We need to do this or git CLI will think we are still merging.
                repo.state_cleanup()
            else:
                raise AssertionError('Unknown merge analysis result')

pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)
repo_path = str(sys.argv[1])
repo = pygit2.Repository(repo_path)
ident = pygit2.Signature('comfyui', 'comfy@ui')
try:
    print("stashing current changes")  # noqa: T201
    repo.stash(ident)
except KeyError:
    print("nothing to stash")  # noqa: T201
except:
    print("Could not stash, cleaning index and trying again.")  # noqa: T201
    repo.state_cleanup()
    repo.index.read_tree(repo.head.peel().tree)
    repo.index.write()
    try:
        repo.stash(ident)
    except KeyError:
        print("nothing to stash.")  # noqa: T201

backup_branch_name = 'backup_branch_{}'.format(datetime.today().strftime('%Y-%m-%d_%H_%M_%S'))
print("creating backup branch: {}".format(backup_branch_name))  # noqa: T201
try:
    repo.branches.local.create(backup_branch_name, repo.head.peel())
except:
    pass

print("checking out master branch")  # noqa: T201
branch = repo.lookup_branch('master')
if branch is None:
    try:
        ref = repo.lookup_reference('refs/remotes/origin/master')
    except:
        print("fetching.")  # noqa: T201
        for remote in repo.remotes:
            if remote.name == "origin":
                remote.fetch()
        ref = repo.lookup_reference('refs/remotes/origin/master')
    repo.checkout(ref)
    branch = repo.lookup_branch('master')
    if branch is None:
        repo.create_branch('master', repo.get(ref.target))
else:
    ref = repo.lookup_reference(branch.name)
    repo.checkout(ref)

print("pulling latest changes")  # noqa: T201
pull(repo)

if "--stable" in sys.argv:
    def latest_tag(repo):
        versions = []
        for k in repo.references:
            try:
                prefix = "refs/tags/v"
                if k.startswith(prefix):
                    version = list(map(int, k[len(prefix):].split(".")))
                    versions.append((version[0] * 10000000000 + version[1] * 100000 + version[2], k))
            except:
                pass
        versions.sort()
        if len(versions) > 0:
            return versions[-1][1]
        return None
    latest_tag = latest_tag(repo)
    if latest_tag is not None:
        repo.checkout(latest_tag)

print("Done!")  # noqa: T201

self_update = True
if len(sys.argv) > 2:
    self_update = '--skip_self_update' not in sys.argv

update_py_path = os.path.realpath(__file__)
repo_update_py_path = os.path.join(repo_path, ".ci/update_windows/update.py")

cur_path = os.path.dirname(update_py_path)


req_path = os.path.join(cur_path, "current_requirements.txt")
repo_req_path = os.path.join(repo_path, "requirements.txt")


def files_equal(file1, file2):
    try:
        return filecmp.cmp(file1, file2, shallow=False)
    except:
        return False

def file_size(f):
    try:
        return os.path.getsize(f)
    except:
        return 0


if self_update and not files_equal(update_py_path, repo_update_py_path) and file_size(repo_update_py_path) > 10:
    shutil.copy(repo_update_py_path, os.path.join(cur_path, "update_new.py"))
    exit()

TORCH_PACKAGES = {'torch', 'torchvision', 'torchaudio'}


def get_installed_torch_versions():
    """Get current versions of torch packages without importing them.

    Returns a dict of {package_name: version_string} for installed torch packages
    that have a platform-specific build variant (e.g. +rocm, +xpu).
    """
    result = {}
    for pkg_name in TORCH_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg_name)
            if spec is None:
                continue
            for folder in spec.submodule_search_locations:
                ver_file = os.path.join(folder, "version.py")
                if os.path.isfile(ver_file):
                    ver_spec = importlib.util.spec_from_file_location(f"{pkg_name}_version", ver_file)
                    module = importlib.util.module_from_spec(ver_spec)
                    ver_spec.loader.exec_module(module)
                    version = module.__version__
                    if '+' in version:
                        result[pkg_name] = version
                    break
        except Exception:
            pass
    return result


def pip_install_requirements(requirements_path):
    """Install requirements while preserving platform-specific torch builds.

    If torch is installed with a non-standard build variant (ROCm, DirectML, XPU, etc.),
    uses pip constraints to pin torch packages and prevent them from being overwritten
    by default PyPI builds.
    """
    import subprocess

    torch_versions = get_installed_torch_versions()
    constraint_file = None

    if torch_versions:
        variant = list(torch_versions.values())[0].split('+')[1]
        print("Detected platform-specific PyTorch build: +{}".format(variant))  # noqa: T201
        print("Constraining torch packages to prevent overwrite: {}".format(  # noqa: T201
            ', '.join('{}=={}'.format(k, v) for k, v in torch_versions.items())
        ))
        constraint_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, prefix='torch_constraints_'
        )
        for pkg, ver in torch_versions.items():
            constraint_file.write("{}=={}\n".format(pkg, ver))
        constraint_file.close()

    try:
        cmd = [sys.executable, '-s', '-m', 'pip', 'install', '-r', requirements_path]
        if constraint_file:
            cmd += ['-c', constraint_file.name]
        subprocess.check_call(cmd)
    finally:
        if constraint_file:
            try:
                os.unlink(constraint_file.name)
            except OSError:
                pass


if not os.path.exists(req_path) or not files_equal(repo_req_path, req_path):
    try:
        pip_install_requirements(repo_req_path)
        shutil.copy(repo_req_path, req_path)
    except:
        pass


stable_update_script = os.path.join(repo_path, ".ci/update_windows/update_comfyui_stable.bat")
stable_update_script_to = os.path.join(cur_path, "update_comfyui_stable.bat")

try:
    if not file_size(stable_update_script_to) > 10:
        shutil.copy(stable_update_script, stable_update_script_to)
except:
    pass

