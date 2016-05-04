# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import requests
from bs4 import BeautifulSoup as BS
from collections import defaultdict


d = defaultdict(list)


def get_topics():
    url = ("http://so.gushiwen.org/shiwen/tags.aspx?"
           "WebShieldDRSessionVerify=UiikVPlyRTdBrTa1uYTR")
    html = requests.get(url).content
    html = BS(html)
    div = html.find_all("div", class_="main3")[0]
    as_ = div.find_all("a")
    topics = [a.text for a in as_]
    return topics


def get_page_urls(topic):
    url = u"http://so.gushiwen.org/type.aspx?p={0}&t={1}"
    return [url.format(str(i+1), topic) for i in range(400)]


def download_topic(topic):
    global d
    urls = get_page_urls(topic)

    def get_poems_urls(url):
        print url
        try:
            html = requests.get(url).content
        except:
            return []
        return BS(html).find_all("a", title="查看全文")

    for url in urls:
        a_s = get_poems_urls(url)
        if not a_s:
            return
        for a in a_s:
            d[a["href"]].append(topic)


    

if __name__ == "__main__":
    topics = get_topics()
    map(download_topic, topics)
    with open("result", "wb") as f:
        import pickle
        pickle.dump(d, f)
