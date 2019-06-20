from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
def page_walker():
    url = "https://www.strd.ru/"
    ki = []
    page = requests.get(url)
    data = page.text
    soup = BeautifulSoup(data,features="lxml")

    for link in soup.find_all('a'):
        ki.append(link.get('href'))
    #delete none
    l = list(filter(None, ki))
    xi =[ x for x in l if "#" not in x ]
    res = [x for x in xi if re.search("https://", x)]
    return res
