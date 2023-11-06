from typing import ClassVar

import scrapy


class MufgSpider(scrapy.Spider):
    name: str = "mufg"
    start_urls: ClassVar[list[str]] = [
        "https://www.am.mufg.jp/mukamapi/fund_search/?site_type=1",
    ]

    def parse(self, response: scrapy.http.Response) -> dict:
        return response.json()["datasets"]["api00001tmCmFndSearchDetailOutDto"]
