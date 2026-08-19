import pytest
from pydantic import ValidationError

from sam_api.config import Settings


def test_quantize_environment_string_is_parsed(monkeypatch):
    monkeypatch.setenv("SAM3_QUANTIZE", "16")
    assert Settings().quantize == 16


def test_invalid_quantize_is_rejected(monkeypatch):
    monkeypatch.setenv("SAM3_QUANTIZE", "8")
    with pytest.raises(ValidationError):
        Settings()
