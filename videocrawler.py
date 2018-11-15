from bs4 import BeautifulSoup
import requests
import urllib
import asyncio

""" taken from curaai00 on Github: https://github.com/curaai00/RT-StyleTransfer-forVideo/blob/master/opticalflow.py """

def get_video(post_url, save_path):
    html = requests.get(post_url).text
    bs = BeautifulSoup(html, 'html.parser')

    # download = bs.find('div', {'class': 'download'})
    links = bs.findAll('a')

    # print(links)

    # print ("LANKSSSSS", [link['href'] for link in links])

    video_link = [link['href'] for link in links if link['href'].endswith('mp4')]

    # print("video", video_link)
    # urllib.request.urlretrieve(video_link, save_path)

    # file_name = video_link.split('/')[-1]

    # create response object

    if (video_link):
        r = requests.get(video_link[0], stream = True)

        print ("Downloading file:%s"%video_link[0])

        # download started
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)

        print(save_path)
        print ("%s downloaded!\n"%video_link[0])

    return save_path

if __name__ == '__main__':
    base_url = "https://www.videvo.net"
    page_url = "https://www.videvo.net/stock-video-footage/video/sort/random/?page=%d"
    save_path = "/Users/StefanHimmel/Desktop/videos/"

    for i in range(3, 30):
        html = requests.get(page_url % i).text
        bs = BeautifulSoup(html, 'html.parser')
        posts = bs.find_all('div', {'class': 'video-responsive columns '})

        print('Downloading %d page now ...' % i)
        # print([base_url + posts[j].find('a')['href'] for j in range(13, len(posts))])
        [get_video(base_url + posts[j].find('a')['href'], save_path + '%d_%d.mp4' % (i, j))  for j in range(13, len(posts))]
