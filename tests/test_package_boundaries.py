import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "packages" / "hyle-core" / "hyle"
_OLLAMA = _ROOT / "packages" / "hyle-ollama" / "hyle_ollama"
_ALLOWED_PRIVATE_CORE_DEPENDENCIES = {"runtime": frozenset({"tools"})}


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def test_core_does_not_depend_on_provider_packages() -> None:
    for path in _CORE.rglob("*.py"):
        assert not any(module.startswith("hyle_ollama") for module in _imports(path))


def test_provider_uses_only_public_core_modules() -> None:
    for path in _OLLAMA.rglob("*.py"):
        private = [module for module in _imports(path) if module.startswith("hyle._")]
        assert private == [], f"{path.relative_to(_ROOT)} imports {private!r}"


def test_private_core_dependencies_follow_one_direction() -> None:
    for path in _CORE.rglob("*.py"):
        relative = path.relative_to(_CORE)
        owner = relative.parts[0] if len(relative.parts) > 1 else None
        allowed = _ALLOWED_PRIVATE_CORE_DEPENDENCIES.get(owner or "", frozenset())

        for module in _imports(path):
            parts = module.split(".")
            if len(parts) < 3 or parts[0] != "hyle" or parts[1] == owner:
                continue
            if not any(part.startswith("_") for part in parts[2:]):
                continue
            assert parts[1] in allowed, (
                f"{path.relative_to(_ROOT)} imports disallowed private sibling {module!r}"
            )
