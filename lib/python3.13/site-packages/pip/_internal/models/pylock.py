import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pip._vendor import tomli_w
from pip._vendor.typing_extensions import Self

from pip._internal.models.direct_url import ArchiveInfo, DirInfo, VcsInfo
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.urls import url_to_path

PYLOCK_FILE_NAME_RE = re.compile(r"^pylock\.([^.]+)\.toml$")


def is_valid_pylock_file_name(path: Path) -> bool:
    return path.name == "pylock.toml" or bool(re.match(PYLOCK_FILE_NAME_RE, path.name))


def _toml_dict_factory(data: List[Tuple[str, Any]]) -> Dict[str, Any]:
    return {key.replace("_", "-"): value for key, value in data if value is not None}


@dataclass
class PackageVcs:
    type: str
    url: Optional[str]
    # (not supported) path: Optional[str]
    requested_revision: Optional[str]
    commit_id: str
    subdirectory: Optional[str]


@dataclass
class PackageDirectory:
    path: str
    editable: Optional[bool]
    subdirectory: Optional[str]


@dataclass
class PackageArchive:
    url: Optional[str]
    # (not supported) path: Optional[str]
    # (not supported) size: Optional[int]
    # (not supported) upload_time: Optional[datetime]
    hashes: Dict[str, str]
    subdirectory: Optional[str]


@dataclass
class PackageSdist:
    name: str
    # (not supported) upload_time: Optional[datetime]
    url: Optional[str]
    # (not supported) path: Optional[str]
    # (not supported) size: Optional[int]
    hashes: Dict[str, str]


@dataclass
class PackageWheel:
    name: str
    # (not supported) upload_time: Optional[datetime]
    url: Optional[str]
    # (not supported) path: Optional[str]
    # (not supported) size: Optional[int]
    hashes: Dict[str, str]


@dataclass
class Package:
    name: str
    version: Optional[str] = None
    # (not supported) marker: Optional[str]
    # (not supported) requires_python: Optional[str]
    # (not supported) dependencies
    vcs: Optional[PackageVcs] = None
    directory: Optional[PackageDirectory] = None
    archive: Optional[PackageArchive] = None
    # (not supported) index: Optional[str]
    sdist: Optional[PackageSdist] = None
    wheels: Optional[List[PackageWheel]] = None
    # (not supported) attestation_identities: Optional[List[Dict[str, Any]]]
    # (not supported) tool: Optional[Dict[str, Any]]

    @classmethod
    def from_install_requirement(cls, ireq: InstallRequirement, base_dir: Path) -> Self:
        base_dir = base_dir.resolve()
        dist = ireq.get_dist()
        download_info = ireq.download_info
        assert download_info
        package = cls(name=dist.canonical_name)
        if ireq.is_direct:
            if isinstance(download_info.info, VcsInfo):
                package.vcs = PackageVcs(
                    type=download_info.info.vcs,
                    url=download_info.url,
                    requested_revision=download_info.info.requested_revision,
                    commit_id=download_info.info.commit_id,
                    subdirectory=download_info.subdirectory,
                )
            elif isinstance(download_info.info, DirInfo):
                package.directory = PackageDirectory(
                    path=(
                        Path(url_to_path(download_info.url))
                        .resolve()
                        .relative_to(base_dir)
                        .as_posix()
                    ),
                    editable=(
                        download_info.info.editable
                        if download_info.info.editable
                        else None
                    ),
                    subdirectory=download_info.subdirectory,
                )
            elif isinstance(download_info.info, ArchiveInfo):
                if not download_info.info.hashes:
                    raise NotImplementedError()
                package.archive = PackageArchive(
                    url=download_info.url,
                    hashes=download_info.info.hashes,
                    subdirectory=download_info.subdirectory,
                )
            else:
                # should never happen
                raise NotImplementedError()
        else:
            package.version = str(dist.version)
            if isinstance(download_info.info, ArchiveInfo):
                if not download_info.info.hashes:
                    raise NotImplementedError()
                link = Link(download_info.url)
                if link.is_wheel:
                    package.wheels = [
                        PackageWheel(
                            name=link.filename,
                            url=download_info.url,
                            hashes=download_info.info.hashes,
                        )
                    ]
                else:
                    package.sdist = PackageSdist(
                        name=link.filename,
                        url=download_info.url,
                        hashes=download_info.info.hashes,
                    )
            else:
                # should never happen
                raise NotImplementedError()
        return package


@dataclass
class Pylock:
    lock_version: str = "1.0"
    # (not supported) environments: Optional[List[str]]
    # (not supported) requires_python: Optional[str]
    # (not supported) extras: List[str] = []
    # (not supported) dependency_groups: List[str] = []
    created_by: str = "pip"
    packages: List[Package] = dataclasses.field(default_factory=list)
    # (not supported) tool: Optional[Dict[str, Any]]

    def as_toml(self) -> str:
        return tomli_w.dumps(dataclasses.asdict(self, dict_factory=_toml_dict_factory))

    @classmethod
    def from_install_requirements(
        cls, install_requirements: Iterable[InstallRequirement], base_dir: Path
    ) -> Self:
        return cls(
            packages=sorted(
                (
                    Package.from_install_requirement(ireq, base_dir)
                    for ireq in install_requirements
                ),
                key=lambda p: p.name,
            )
        )
