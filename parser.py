import urllib.request
from bs4 import BeautifulSoup, SoupStrainer
import ssl
import requests
from site_walker import page_walker
import random
from time import gmtime, strftime

def open_page(link):

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(link,context = ctx) as response:
        html = response.read()
    return html

def find_images(page):
    imge = []
    soup = BeautifulSoup(page)
    for img in soup.findAll('img'):
        imge.append(img.get('src'))
    return imge

def create_name_folder():
    links = page_walker()
    names = list()
    for i in links:
        names.append(str(i).split('/')[-2])
    return names
def download_stuff(img_link):
    domen ='https://www.strd.ru/'
    print('error here')
    ssl._create_default_https_context = ssl._create_unverified_context
    for link in img_link:
        url = domen+link

        number =random.randint(1,1000)
        name =strftime("%Y-%m-%d")+"-"+str(number)
        r = requests.get(str(url))
        with open('img_folder/'+str(name)+'.jpg', 'wb') as f:
            print("{}.jpg created".format(name))
            f.write(r.content)
            f.close()

def take_pictures_from_one_page(link):
        page = open_page(link)
        img_link = find_images(page)
        download_stuff(img_link)


def gohead_to_site():



def main():
    print(create_links())
    print("mission complete")


if __name__ == "__main__":
    main()

