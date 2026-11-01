"""Discover published SDV assets and retain verified original bytes."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import pyarrow.parquet as pq

CONFIG = Path(__file__).resolve().parents[1] / "config" / "scouting_sources.json"

class SourceUnavailable(RuntimeError):
    """A required asset has not been published."""

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

class HTTPClient:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    def _open(self, url, headers=None):
        if urlparse(url).scheme != "https":
            raise ValueError("Only HTTPS source requests are supported")
        selected = {"User-Agent": "wbb-bis-scouting/0.1", **(headers or {})}
        if self.token and urlparse(url).hostname == "api.github.com":
            selected["Authorization"] = "Bearer " + self.token
        for attempt in range(3):
            try:
                return urlopen(Request(url, headers=selected), timeout=40)
            except HTTPError as exc:
                if exc.code == 404:
                    raise SourceUnavailable("Published asset is unavailable") from exc
                if exc.code == 304:
                    return exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("Request retry exhausted")
    def get_json(self, url):
        with self._open(url, {"Accept":"application/vnd.github+json"}) as response:
            return json.load(response)
    def download(self, url, path, headers=None):
        with self._open(url, headers) as response:
            status = response.status
            selected = dict(response.headers)
            if status != 304:
                with path.open("wb") as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            return status, selected

def source_config() -> dict:
    return json.loads(CONFIG.read_text())

def discover_assets(dataset: str, season: int, client: object) -> list[dict]:
    config = source_config()
    spec = config["datasets"][dataset]
    url = "https://api.github.com/repos/" + config["repository"] + "/releases/tags/" + spec["tag"]
    try:
        release = client.get_json(url)
    except SourceUnavailable:
        return []
    target = spec["pattern"].format(season=season)
    matches = [dict(asset) for asset in release.get("assets", []) if asset["name"] == target]
    if len(matches) > 1:
        raise ValueError("Conflicting published asset names")
    for asset in matches:
        asset.update(dataset=dataset, season=season, family=config["family"],
                     league=config["league"], tag=spec["tag"], provider="SportsDataverse")
    return matches

def download_asset(asset: dict, destination: Path, client: object) -> dict:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = asset["browser_download_url"]
    prefix = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    if not url.startswith(prefix):
        raise ValueError("Unexpected SDV asset source")
    expected = (asset.get("digest") or "").removeprefix("sha256:")
    if expected and (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected.lower())):
        raise ValueError("Invalid published checksum")
    reused = destination.exists() and bool(expected) and sha256_file(destination) == expected
    temporary = destination.with_name(destination.name + ".part")
    try:
        if not reused:
            headers = {"If-None-Match": asset["cached_etag"]} if asset.get("cached_etag") and destination.exists() else {}
            status, response_headers = client.download(url, temporary, headers)
            if status == 304:
                if not destination.exists():
                    raise ValueError("HTTP 304 without a verified previous file")
                reused = True
            elif status == 200:
                pq.ParquetFile(temporary)
                if expected and sha256_file(temporary) != expected:
                    raise ValueError("Published checksum mismatch")
                temporary.replace(destination)
            else:
                raise ValueError("Unexpected download HTTP status")
        parquet = pq.ParquetFile(destination)
        checksum = sha256_file(destination)
        if expected and checksum != expected:
            raise ValueError("Published checksum mismatch")
        return {**asset, "sha256":checksum, "bytes":destination.stat().st_size,
                "row_count":parquet.metadata.num_rows,
                "schema_fingerprint":hashlib.sha256(str(parquet.schema_arrow).encode()).hexdigest(),
                "columns":parquet.schema_arrow.names, "local_path":str(destination), "reused":reused}
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        if not isinstance(exc, (ValueError, SourceUnavailable, HTTPError, URLError)):
            raise ValueError("Unreadable Parquet asset") from exc
        raise
