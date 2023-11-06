# Portfolio Optimizer

How to run

```
rm -f crawler/mufg_funds/*.json
cd crawler
rye sync
cd mufg_funds
rye run scrapy crawl mufg -o fundlist.json
rye run scrapy crawl mufg_chart -o fundchart.json

cd ../../web
rye sync
rye run streamlit run src/app.py --server.port 8502
```
