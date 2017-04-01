'''
InstagramML - Unpossibly API Wrapper
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import requests as req

class API:
    def __init__(self, username, api_key):
        self.username = username
        self.api_key = api_key
        self.BASE_URL = 'http://'+self.username+':'+self.api_key+'@challenges.instagram.unpossib.ly/api/'

    def get_posts(self):
        r = req.get(self.BASE_URL+'live')
        return r.json()

    def submit(self, post, likes):
        payload = {'post': post, 'likes': likes}
        r = req.post(self.BASE_URL+'submissions', json=payload)
        return r.json()

    def review(self):
        r = req.get(self.BASE_URL+'submissions')
        return r.json()

if __name__ == '__main__':
    username = 'yuyajeremyong@gmail.com'
    key = '6g6wDQF5sceQtEZeUKJWYx0o6Uer4vGg'
    api = API(username, key)
    print api.get_posts()
