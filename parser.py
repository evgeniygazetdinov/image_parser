import urllib.request
from bs4 import BeautifulSoup, SoupStrainer
import ssl
import requests
import random
from time import gmtime, strftime

from create_and_move import File_manager
from site_walker import page_walker


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
    domen ='https://www.strd.ru'

    print(img_link)
    counter = 0
    ssl._create_default_https_context = ssl._create_unverified_context
    for link in img_link:
        #TODO add some delay wainting create file


        url = domen+link
        print(url)
        number =link.split('/')
        name =strftime("%Y-%m-%d")+"-"+str(number[-1])
        r = requests.get(str(url))
        with open('img_folder/'+str(name), 'wb') as f:
            print("{}.jpg created".format(name))
            counter=+1
            print(counter)
            f.write(r.content)
        f.close()


def take_pictures_from_one_page(link,name):
        f = File_manager()
        page = open_page(link)
        img_link = find_images(page)
        download_stuff(img_link)





def gohead_to_site():
    name_folder = create_name_folder()
    links = page_walker()
    counter=1
    for link in links:
        take_pictures_from_one_page(link,str(counter))
        links.remove(link)

def main():
    gohead_to_site()
    print("mission complete")


if __name__ == "__main__":
    main()

