from __future__ import annotations

import base64
import hashlib
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import cache
from typing import (
    Protocol,
)


@dataclass
class File:
    name: str
    mime_type: str
    data: bytes
    metadata: dict[str, str] | None = field(default=None)

    @property
    @cache
    def content_md5(self):
        """RFC1864: base64-encoded MD5 digests"""
        return base64.b64encode(hashlib.md5(self.data).digest())


class ArtifactStorage(Protocol):
    async def exists(self, url: str) -> bool: ...
    async def download(self, url: str) -> File: ...
    async def upload(self, file: File) -> str: ...


_context_artifact_repository = ContextVar[ArtifactStorage]("artifact_storage")


def get_artifact_storage() -> ArtifactStorage:
    return _context_artifact_repository.get()


def set_artifact_storage(repo: ArtifactStorage) -> Token[ArtifactStorage]:
    return _context_artifact_repository.set(repo)


def reset_artifact_storage(token: Token[ArtifactStorage]) -> None:
    _context_artifact_repository.reset(token)


async def upload_artifact(file: File) -> str:
    return await get_artifact_storage().upload(file)


async def download_artifact(url: str) -> File:
    return await get_artifact_storage().download(url)
