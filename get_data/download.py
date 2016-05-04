# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
import pickle
import requests
from bs4 import BeautifulSoup as bs
import re

with open("result", "rb") as f:
    d = pickle.load(f)


def download(url):
    global d
    print url
    html = requests.get(get_url(url)).content
    soup = bs(html)
    main3 = soup.find_all("div", class_="main3")[0]
    son2 = main3.find_all("div", class_="son2")[0]
    son1 = main3.find_all("div", class_="son1")[0]
    r = {}
    r['title'] = son1.text.strip()
    r['content'] = u"".join(son2.text.split()[5:])
    pattern = re.compile(ur"\(.*\)", re.UNICODE)
    r['content'] = re.sub(pattern, "", r['content'])
    pattern = re.compile(ur"\（.*\）", re.UNICODE)
    r['content'] = re.sub(pattern, "", r['content'])
    r['topics'] = d[url]
    print r['title']
    print r['content']
    return r


def get_url(url):
    return "http://so.gushiwen.org" + url



if __name__ == "__main__":
    global d
    result = [download(url) for url in d]
    with open("result_poem","wb") as f:
        pickle.dump(result, f)
