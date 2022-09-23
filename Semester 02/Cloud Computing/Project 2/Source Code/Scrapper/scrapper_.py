import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as rq
from tqdm import tqdm
import numpy as np
import boto3


class Scrapper:

    def __init__(self):
        self.page_url = 'https://www.otomoto.pl/motocykle-i-quady'

    def find_no_pages(self):
        page = self.get_page(self.page_url)
        soup = BeautifulSoup(page, 'html.parser')
        n_of_page = soup.select("li.pagination-item > a > span")[-1].text
        return n_of_page

    def get_page(self, page):
        return rq.urlopen(page)

    def get_links(self):
        n_of_page = self.find_no_pages()
        links_ = []
        # for i in range(2, int(n_of_page), 1):
        for i in range(2, 400, 1):
            # time.sleep(1)
            page_url = f"{self.page_url}?page={i}"
            page = rq.urlopen(page_url)
            soup = BeautifulSoup(page, 'html.parser')
            link = [link['href'] for link in soup.select("main > article > div > h2 > a[href*='otomoto.pl']")]
            links_.append(link)
        return [i for links in links_ for i in links]

    def main(self):
        links = self.get_links()
        df = pd.DataFrame()

        for link in tqdm(links):
            try:
                page = self.get_page(link)
                page = BeautifulSoup(page, 'html.parser')
                items = [i.text for i in page.find_all("li", class_="offer-params__item")]
                items = [item.replace("\n", "").replace("              ", "").split("  ")[0:2] for item in items]

                price = page.find_all("span", class_="offer-price__number")[0].text
                price = "".join([digit for digit in price if digit.isdigit()])

                dict_ = {item[0]: item[1] for item in items}
                dict_['price'] = price
                df = df.append(dict_, ignore_index=True)
                print("Data Has been corectly scrapped")
            except Exception as E:
                print(E)

        return df


def export(df, no_of_rows=50):
    for k, g in df.groupby(np.arange(len(df)) // no_of_rows):
        g.to_json('files/Export{}.json'.format(k + 1), orient='records')

    return k


def push_to_s3(k):
    print("pushing to s3")
    [s3.upload_file('/home/ubuntu/scrapper/files/Export{}.json'.format(i + 1), 'scrapperbucketcc', f'Export{i + 1}.json') for i in range(k + 1)]


df = Scrapper().main()
#k = export(df)
no_of_rows = 50
for k, g in df.groupby(np.arange(len(df)) // no_of_rows):
    g.to_json('files/Export{}.json'.format(k + 1), orient='records')

#s3 = boto3.client('s3')
#push_to_s3(k)

#
# df.to_json("files/scrapped_23052022.json")


