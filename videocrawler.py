from bs4 import BeautifulSoup
import requests
import urllib
import asyncio

""" taken from curaai00 on Github: https://github.com/curaai00/RT-StyleTransfer-forVideo/blob/master/opticalflow.py """

def get_video(post_url, save_path):
    html = requests.get(post_url).text
    bs = BeautifulSoup(html, 'html.parser')

    download = bs.find('div', {'class': 'download'})
    video = download.find('a')['href']
    urllib.request.urlretrieve(video, save_path)

    return save_path

if __name__ == '__main__':
    base_url = "https://www.videvo.net"
    page_url = "https://www.videvo.net/stock-video-footage/video/sort/random/?page=%d"
    save_path = "video/"

    for i in range(3, 30):
        html = requests.get(page_url % i).text
        bs = BeautifulSoup(html, 'html.parser')
        posts = bs.find_all('div', {'class': 'video-responsive columns '})

        print('Downloading %d page now ...' % i)
        [get_video(base_url + posts[j].find('a')['href'], save_path + '%d_%d.mp4' % (i, j))  for j in range(13, len(posts))]
