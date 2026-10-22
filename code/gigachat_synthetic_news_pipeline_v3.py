"""v3 launcher: bind all GigaChat HTTPS requests to the physical Wi-Fi NIC."""

from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

import gigachat_synthetic_news_pipeline_v1 as v1
import gigachat_synthetic_news_pipeline_v2 as v2


class SourceAddressAdapter(HTTPAdapter):
    def __init__(self, source_address: str, **kwargs):
        self.source_address = source_address
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["source_address"] = (self.source_address, 0)
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **pool_kwargs)


_original_init = v1.GigaChatClient.__init__


def _wifi_init(self) -> None:
    _original_init(self)
    source = os.getenv("GIGACHAT_SOURCE_ADDRESS", "192.168.0.12")
    self.session.mount("https://", SourceAddressAdapter(source))


v1.GigaChatClient.__init__ = _wifi_init


if __name__ == "__main__":
    v2.main()
