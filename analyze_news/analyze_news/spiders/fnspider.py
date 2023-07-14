import scrapy
from scrapy.selector import Selector
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time


class FoxnewsSpider(scrapy.Spider):
    name = 'foxnews'
    allowed_domains = ['www.foxnews.com']
    start_urls = ['https://www.foxnews.com']

    def __init__(self):
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(executable_path=str('./chromedriver'), options=chrome_options)
        driver.get("https://www.foxnews.com/category/us/immigration")

        wait = WebDriverWait(driver, 10)

        # first, click 'Show More' many times
        i = 0
        while i < 80:
            try:
                time.sleep(1)
                element = wait.until(EC.visibility_of_element_located(
                    (By.XPATH, "(//div[@class='button load-more js-load-more'])[1]/a")))
                element.click()
                i += 1
            except TimeoutException:
                break

        # then, copy down all that's now shown on the page
        self.html = driver.page_source

    def parse(self, response):
        resp = Selector(text=self.html)
        results = resp.xpath("//article[@class='article']//h4[@class='title']/a")
        for result in results:
            title = result.xpath(".//text()").get()
            eyebrow = result.xpath(".//span[@class='eyebrow']/a/text()").get()  # scraped only for filtering
            link = result.xpath(".//@href").get()
            if eyebrow == 'VIDEO':
                continue  # filter out videos
            else:
                yield response.follow(url=link, callback=self.parse_article, meta={"title": title})

    def parse_article(self, response):
        title = response.request.meta['title']
        authors = response.xpath("(//div[@class='author-byline']//span/a)[1]/text()").getall()
        if len(authors) == 0:
            authors = [i for i in response.xpath("//div[@class='author-byline opinion']//span/a/text()").getall() if
                       'Fox News' not in i]
        content = ' '.join(response.xpath("//div[@class='article-body']//text()").getall())
        yield {
            "title": title,
            "byline": ' '.join(authors),
            "time": response.xpath("//div[@class='article-date']/time/text()").get(),
            "content": content
        }