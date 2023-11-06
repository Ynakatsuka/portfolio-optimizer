import json
from pathlib import Path
from typing import ClassVar

import scrapy


def load_fund_codes(filename: str) -> list[str]:
    with Path(filename).open() as file:
        data = json.load(file)
        return [item["cfsd_fund_cd"] for item in data]


class MufgChartSpider(scrapy.Spider):
    name: str = "mufg_chart"
    allowed_domains: ClassVar[list[str]] = ["www.am.mufg.jp"]

    def start_requests(self) -> scrapy.Request:
        fund_codes = load_fund_codes("fundlist.json")
        for fund_code in fund_codes:
            url = f"https://www.am.mufg.jp/fund_file/chart/chart_data_{fund_code}.js"
            yield scrapy.Request(url, self.parse)

    def parse(self, response: scrapy.http.Response) -> dict:
        return response.json()
