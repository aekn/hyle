import argparse
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from tarfile import TarFile
from tarfile import open as open_tar
from typing import TypeIs
from zipfile import ZipFile


@dataclass(frozen=True, slots=True)
class Package:
    name: str
    version: str
    requires_python: str
    license: str
    import_names: tuple[str, ...]
    dependencies: tuple[str, ...]
    license_files: tuple[Path, ...]
    root: Path
    module: str
    module_root: Path
    readme: Path
    source_includes: tuple[str, ...]

    @property
    def module_dir(self) -> Path:
        return self.root / self.module_root / Path(*self.module.split("."))

    @property
    def module_archive_path(self) -> Path:
        return self.module_root / Path(*self.module.split("."))


@dataclass(frozen=True, slots=True)
class Metadata:
    name: str
    version: str
    requires_python: str
    license: str
    import_names: tuple[str, ...]
    license_files: tuple[str, ...]
    dependencies: tuple[str, ...]


_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="verify Hyle distribution artifacts")
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        required=True,
        help="distribution to verify; repeat for multiple packages",
    )
    parser.add_argument("artifact", nargs="+", type=Path)
    arguments = parser.parse_args()

    packages = _load_selected_packages(arguments.packages)
    verified = {name: set[str]() for name in packages}

    for artifact in arguments.artifact:
        if artifact.suffix == ".whl":
            name = _verify_wheel(artifact, packages)
            kind = "wheel"
        elif artifact.name.endswith(".tar.gz"):
            name = _verify_sdist(artifact, packages)
            kind = "sdist"
        else:
            raise SystemExit(f"unsupported distribution artifact: {artifact}")

        if kind in verified[name]:
            raise SystemExit(f"duplicate {kind} for {name!r}")
        verified[name].add(kind)
        print(f"verified {name} {kind}: {artifact}")

    missing = {
        name: {"wheel", "sdist"} - kinds
        for name, kinds in verified.items()
        if kinds != {"wheel", "sdist"}
    }
    if missing:
        details = ", ".join(
            f"{name}: {', '.join(sorted(kinds))}" for name, kinds in sorted(missing.items())
        )
        raise SystemExit(f"missing distribution artifacts ({details})")


def _load_selected_packages(names: Sequence[str], /) -> dict[str, Package]:
    selected = tuple(names)
    if len(selected) != len(set(selected)):
        raise SystemExit("each --package value must be unique")

    roots_by_name: dict[str, Path] = {}
    for root in _workspace_roots():
        pyproject = root / "pyproject.toml"
        config = _load_toml(pyproject)
        project = _table(config, "project", pyproject)
        name = _string(project, "name", pyproject)
        if name in roots_by_name:
            raise SystemExit(
                f"duplicate workspace project {name!r}: {roots_by_name[name]} and {root}"
            )
        roots_by_name[name] = root

    missing = set(selected) - roots_by_name.keys()
    if missing:
        formatted = ", ".join(repr(name) for name in sorted(missing))
        raise SystemExit(f"unknown workspace package: {formatted}")

    return {name: _load_package(roots_by_name[name]) for name in selected}


def _workspace_roots() -> tuple[Path, ...]:
    pyproject = _ROOT / "pyproject.toml"
    root_config = _load_toml(pyproject)
    tool = _table(root_config, "tool", pyproject)
    uv = _table(tool, "uv", pyproject)
    workspace = _table(uv, "workspace", pyproject)
    patterns = _strings(workspace, "members", pyproject, required=True)

    roots: set[Path] = set()
    for pattern in patterns:
        matches = tuple(path for path in sorted(_ROOT.glob(pattern)) if path.is_dir())
        if not matches:
            raise SystemExit(f"workspace member pattern {pattern!r} matched no directories")
        roots.update(matches)
    return tuple(sorted(roots))


def _load_package(root: Path) -> Package:
    pyproject = root / "pyproject.toml"
    config = _load_toml(pyproject)
    project = _table(config, "project", pyproject)
    build_system = _table(config, "build-system", pyproject)
    if _string(build_system, "build-backend", pyproject) != "uv_build":
        raise SystemExit(f"{pyproject} must use the uv_build backend")

    tool = _table(config, "tool", pyproject)
    uv = _table(tool, "uv", pyproject)
    build = _table(uv, "build-backend", pyproject)
    license_patterns = _strings(project, "license-files", pyproject, required=True)

    package = Package(
        name=_string(project, "name", pyproject),
        version=_string(project, "version", pyproject),
        requires_python=_string(project, "requires-python", pyproject),
        license=_string(project, "license", pyproject),
        import_names=_strings(project, "import-names", pyproject, required=True),
        dependencies=_strings(project, "dependencies", pyproject),
        license_files=_matched_files(root, license_patterns, "license-files"),
        root=root,
        module=_string(build, "module-name", pyproject),
        module_root=Path(_module_root(build, pyproject)),
        readme=Path(_readme(project, pyproject)),
        source_includes=_strings(build, "source-include", pyproject),
    )
    _validate_source(package)
    return package


def _validate_source(package: Package) -> None:
    if package.import_names != (package.module,):
        raise SystemExit(f"{package.name!r} must declare import-names = [{package.module!r}]")
    if not package.module_dir.is_dir():
        raise SystemExit(f"module directory does not exist: {package.module_dir}")
    if not (package.module_dir / "__init__.py").is_file():
        raise SystemExit(f"module {package.module!r} has no __init__.py")
    if not (package.module_dir / "py.typed").is_file():
        raise SystemExit(f"typed package {package.module!r} is missing py.typed")
    if not (package.root / package.readme).is_file():
        raise SystemExit(f"readme does not exist: {package.root / package.readme}")
    if not package.license_files:
        raise SystemExit(f"{package.name!r} has no matched license files")


def _verify_wheel(wheel: Path, packages: dict[str, Package]) -> str:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = _only(names, ".dist-info/METADATA", wheel)
        metadata = _metadata(archive.read(metadata_name), wheel)
        package = _match_package(metadata, wheel, packages)
        module_prefix = Path(*package.module.split("."))

        expected = {
            (module_prefix / path.relative_to(package.module_dir)).as_posix()
            for path in _files(package.module_dir)
        }
        _require_files(wheel, expected, names, "package")

        dist_info = Path(metadata_name).parent
        license_expected = {
            (dist_info / "licenses" / path.relative_to(package.root)).as_posix()
            for path in package.license_files
        }
        _require_files(wheel, license_expected, names, "license")
        _reject_files(wheel, {name for name in names if _forbidden_wheel(name)})
        return package.name


def _verify_sdist(sdist: Path, packages: dict[str, Package]) -> str:
    with open_tar(sdist, mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        metadata_name = _only(names, "/PKG-INFO", sdist)
        metadata = _metadata(_read_tar(archive, metadata_name, sdist), sdist)
        package = _match_package(metadata, sdist, packages)
        prefix = Path(metadata_name).parent

        expected = {
            (prefix / "pyproject.toml").as_posix(),
            (prefix / package.readme).as_posix(),
            *(
                (
                    prefix / package.module_archive_path / path.relative_to(package.module_dir)
                ).as_posix()
                for path in _files(package.module_dir)
            ),
            *(
                (prefix / path.relative_to(package.root)).as_posix()
                for path in package.license_files
            ),
            *_source_includes(package, prefix),
        }
        _require_files(sdist, expected, names, "source")
        _reject_files(sdist, {name for name in names if _forbidden_sdist(name)})
        return package.name


def _metadata(data: bytes, artifact: Path) -> Metadata:
    message = BytesParser(policy=compat32).parsebytes(data)
    return Metadata(
        name=_field(message, "Name", artifact),
        version=_field(message, "Version", artifact),
        requires_python=_field(message, "Requires-Python", artifact),
        license=_field(message, "License-Expression", artifact),
        import_names=_fields(message, "Import-Name"),
        license_files=_fields(message, "License-File"),
        dependencies=_fields(message, "Requires-Dist"),
    )


def _match_package(
    metadata: Metadata,
    artifact: Path,
    packages: dict[str, Package],
) -> Package:
    package = packages.get(metadata.name)
    if package is None:
        raise SystemExit(f"unexpected distribution {metadata.name!r} in {artifact}")

    checks = (
        ("Version", metadata.version, package.version),
        ("Requires-Python", metadata.requires_python, package.requires_python),
        ("License-Expression", metadata.license, package.license),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise SystemExit(f"{artifact} has {field} {actual!r}; expected {expected!r}")

    if metadata.import_names != package.import_names:
        raise SystemExit(
            f"{artifact} has Import-Name {metadata.import_names!r}; "
            f"expected {package.import_names!r}"
        )

    expected_licenses = tuple(
        path.relative_to(package.root).as_posix() for path in package.license_files
    )
    if metadata.license_files != expected_licenses:
        raise SystemExit(
            f"{artifact} has License-File {metadata.license_files!r}; "
            f"expected {expected_licenses!r}"
        )

    if set(metadata.dependencies) != set(package.dependencies):
        raise SystemExit(
            f"{artifact} has Requires-Dist {metadata.dependencies!r}; "
            f"expected {package.dependencies!r}"
        )
    return package


def _field(message: Message, name: str, artifact: Path) -> str:
    value = message.get(name)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{artifact} metadata does not contain {name}")
    return value


def _fields(message: Message, name: str) -> tuple[str, ...]:
    values = message.get_all(name, [])
    return tuple(values)


def _source_includes(package: Package, prefix: Path) -> set[str]:
    included: set[str] = set()
    for pattern in package.source_includes:
        for match in package.root.glob(pattern):
            paths = (match,) if match.is_file() else _files(match) if match.is_dir() else ()
            for path in paths:
                relative = path.relative_to(package.root)
                if not _forbidden(relative):
                    included.add((prefix / relative).as_posix())
    return included


def _matched_files(root: Path, patterns: tuple[str, ...], field: str) -> tuple[Path, ...]:
    matches = {
        path
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file() and not _forbidden(path.relative_to(root))
    }
    if not matches:
        raise SystemExit(f"{root / 'pyproject.toml'} field {field!r} matched no files")
    return tuple(sorted(matches))


def _files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _forbidden(path.relative_to(root))
    )


def _only(names: set[str], suffix: str, artifact: Path) -> str:
    matches = tuple(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        raise SystemExit(f"{artifact} contains {len(matches)} files ending with {suffix!r}")
    return matches[0]


def _read_tar(archive: TarFile, name: str, artifact: Path) -> bytes:
    file = archive.extractfile(name)
    if file is None:
        raise SystemExit(f"could not read {name!r} from {artifact}")
    return file.read()


def _forbidden(path: Path) -> bool:
    return (
        "__pycache__" in path.parts or ".DS_Store" in path.parts or path.suffix in {".pyc", ".pyo"}
    )


def _forbidden_wheel(name: str) -> bool:
    path = Path(name)
    return _forbidden(path) or "tests" in path.parts or "examples" in path.parts


def _forbidden_sdist(name: str) -> bool:
    path = Path(name)
    return _forbidden(path) or any(
        part in {".git", ".venv", ".pytest_cache", ".ruff_cache", ".hypothesis", "dist", "build"}
        for part in path.parts
    )


def _load_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"configuration file does not exist: {path}")
    with path.open("rb") as file:
        value: object = tomllib.load(file)
    return _as_table(value, path)


def _table(data: dict[str, object], key: str, source: Path) -> dict[str, object]:
    return _as_table(data.get(key), source, field=key)


def _as_table(
    value: object,
    source: Path,
    /,
    *,
    field: str | None = None,
) -> dict[str, object]:
    if not _is_mapping(value):
        suffix = "" if field is None else f" {field!r}"
        raise SystemExit(f"{source} must define TOML table{suffix}")
    table: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SystemExit(f"{source} contains a non-string TOML key")
        table[key] = item
    return table


def _string(data: dict[str, object], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{source} must define non-empty string {key!r}")
    return value


def _module_root(build: dict[str, object], source: Path) -> str:
    value = build.get("module-root", "src")
    if not isinstance(value, str):
        raise SystemExit(f"{source} field 'module-root' must be a string")
    return value


def _strings(
    data: dict[str, object],
    key: str,
    source: Path,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    value = data.get(key, ())
    if not _is_sequence(value):
        raise SystemExit(f"{source} field {key!r} must be an array of strings")

    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SystemExit(f"{source} field {key!r} must be an array of strings")
        items.append(item)

    if required and not items:
        raise SystemExit(f"{source} must define at least one value in {key!r}")
    return tuple(items)


def _readme(project: dict[str, object], source: Path) -> str:
    value = project.get("readme")
    if isinstance(value, str) and value:
        return value
    if _is_mapping(value):
        table = _as_table(value, source, field="project.readme")
        file = table.get("file")
        if isinstance(file, str) and file:
            return file
    raise SystemExit(f"{source} must define a file-backed project readme")


def _is_mapping(value: object, /) -> TypeIs[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_sequence(value: object, /) -> TypeIs[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    )


def _require_files(
    artifact: Path,
    expected: set[str],
    actual: set[str],
    kind: str,
) -> None:
    missing = expected - actual
    if missing:
        formatted = "\n".join(f"  - {name}" for name in sorted(missing))
        raise SystemExit(f"{artifact} is missing {kind} files:\n{formatted}")


def _reject_files(artifact: Path, forbidden: set[str]) -> None:
    if forbidden:
        formatted = "\n".join(f"  - {name}" for name in sorted(forbidden))
        raise SystemExit(f"{artifact} contains forbidden files:\n{formatted}")


if __name__ == "__main__":
    main()
