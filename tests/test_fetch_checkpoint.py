import io
import urllib.error

import pytest

from anna_inference import fetch_checkpoint


def _http_error(code):
    return urllib.error.HTTPError("url", code, "error", {}, None)


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(fetch_checkpoint.time, "sleep", lambda s: None)


def _urlopen(responses):
    calls = []

    def urlopen(url):
        calls.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(response)

    return urlopen, calls


def test_download_retries_server_errors(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(504), b"checkpoint"])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    fp = tmp_path / "ckpt"
    fetch_checkpoint._download("url", fp)
    assert fp.read_bytes() == b"checkpoint"
    assert len(calls) == 2


def test_download_gives_up(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(504)] * 3)
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        fetch_checkpoint._download("url", tmp_path / "ckpt", attempts=3)
    assert len(calls) == 3


def test_download_no_retry_on_client_error(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(404), b"checkpoint"])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        fetch_checkpoint._download("url", tmp_path / "ckpt")
    assert len(calls) == 1
