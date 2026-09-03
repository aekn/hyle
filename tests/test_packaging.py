from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "packages" / "hyle-core"
_OLLAMA = _ROOT / "packages" / "hyle-ollama"


def test_core_distribution_name_is_distinct_from_import_name() -> None:
    pyproject = (_CORE / "pyproject.toml").read_text()

    assert 'name = "hyle-core"' in pyproject
    assert 'import-names = ["hyle"]' in pyproject
    assert 'module-name = "hyle"' in pyproject
    assert (_CORE / "hyle" / "__init__.py").is_file()
    assert not (_CORE / "hyle_core").exists()


def test_provider_tracks_core_alpha_exactly() -> None:
    core = (_CORE / "pyproject.toml").read_text()
    provider = (_OLLAMA / "pyproject.toml").read_text()

    assert 'version = "0.1.0a1"' in core
    assert 'version = "0.1.0a1"' in provider
    assert 'name = "hyle-ollama"' in provider
    assert 'import-names = ["hyle_ollama"]' in provider
    assert '"hyle-core==0.1.0a1"' in provider


def test_package_licenses_match_repository_license() -> None:
    license_text = (_ROOT / "LICENSE").read_bytes()
    assert (_CORE / "LICENSE").read_bytes() == license_text
    assert (_OLLAMA / "LICENSE").read_bytes() == license_text
