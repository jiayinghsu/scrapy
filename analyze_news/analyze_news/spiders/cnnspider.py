import scrapy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from scrapy.selector import Selector
import time


class CnnSpider(scrapy.Spider):
    name = 'cnn'
    allowed_domains = ['www.cnn.com']
    start_urls = ['https://www.cnn.com']

    # initiating selenium
    def __init__(self):

        # set up the driver
        chrome_options = Options()
        # chrome_options.add_argument("--headless") # uncomment if don't want to appreciate the sight of a possessed browser
        driver = webdriver.Chrome(executable_path=str('./chromedriver'), options=chrome_options)
        driver.get("https://www.cnn.com")

        # begin search
        search_input = driver.find_element_by_id("footer-search-bar")  # find the search bar
        search_input.send_keys("immigration")  # type in the search term
        search_btn = driver.find_element_by_xpath(
            "(//button[contains(@class, 'Flex-sc-1')])[2]")  # find the search button
        search_btn.click()

        # record the first page
        self.html = [driver.page_source]

        # start turning pages
        i = 0
        while i < 100:  # 100 is just right to get us back to July
            i += 1
            time.sleep(5)  # just in case the next button hasn't finished loading
            next_btn = driver.find_element_by_xpath(
                "(//div[contains(@class, 'pagination-arrow')])[2]")  # click next button
            next_btn.click()
            self.html.append(driver.page_source)  # not the best way but will do

    # using scrapy's native parse to first scrape links on result pages
    def parse(self, response):
        for page in self.html:
            resp = Selector(text=page)
            results = resp.xpath(
                "//div[@class='cnn-search__result cnn-search__result--article']/div/h3/a")  # result iterator
            for result in results:
                title = result.xpath(".//text()").get()
                if ("Video" in title) | ("coronavirus news" in title) | ("http" in title):
                    continue  # ignore videos and search-independent news or ads
                else:
                    link = result.xpath(".//@href").get()[
                           13:]  # cut off the domain; had better just use request in fact
                    yield response.follow(url=link, callback=self.parse_article, meta={"title": title})

    # pass on the links to open and process actual news articles
    def parse_article(self, response):
        title = response.request.meta['title']

        # several variations of author's locator
        authors = response.xpath("//span[@class='metadata__byline__author']//text()").getall()
        if len(authors) == 0:
            authors = response.xpath("//p[@data-type='byline-area']//text()").getall()
            if len(authors) == 0:
                authors = response.xpath("//div[@class='Article__subtitle']//text()").getall()

        # two variations of article body's locator
        content = ' '.join(response.xpath("//section[@id='body-text']/div[@class='l-container']//text()").getall())
        if content is None:
            content = ' '.join(response.xpath("//div[@class='Article__content']//text()").getall())
        yield {
            "title": title,
            "byline": ' '.join(authors),  # could be multiple authors
            "time": response.xpath("//p[@class='update-time']/text()").get(),
            "content": content
        }