import base64
import hashlib
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import cache
from typing import Protocol


@dataclass(frozen=True)
class Blob:
    mime_type: str
    data: bytes

    @property
    @cache
    def content_md5(self):
        """RFC1864: base64-encoded MD5 digests"""
        return base64.b64encode(hashlib.md5(self.data).digest())


class BlobStorage(Protocol):
    async def exists(self, url: str) -> bool: ...
    async def download(self, url: str) -> Blob: ...
    async def upload(self, blob: Blob) -> str: ...


_context_blob_resolver = ContextVar[BlobStorage]("blob_storage")


def get_blob_storage() -> BlobStorage:
    return _context_blob_resolver.get()


def set_blob_storage(resolver: BlobStorage) -> Token[BlobStorage]:
    return _context_blob_resolver.set(resolver)


def reset_blob_storage(token: Token[BlobStorage]) -> None:
    _context_blob_resolver.reset(token)


async def upload_blob(blob: Blob) -> str:
    return await get_blob_storage().upload(blob)


async def download_blob(url: str) -> Blob:
    return await get_blob_storage().download(url)
