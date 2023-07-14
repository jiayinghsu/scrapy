import scrapy


class CrawlSpider(scrapy.Spider):
    name = 'crawl'
    allowed_domains = ['news']
    start_urls = ['http://news/']

    def parse(self, response):
        pass
