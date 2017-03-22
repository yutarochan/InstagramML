'''
InstagramML - Data Parser & Preprocessing
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import json

def parse_json(filename):
    json_file = open(filename, 'rb')
    return json.loads(json_file.readlines()[0])

parse_json('../data/dataset.json')[0]['username']
parse_json('../data/dataset.json')[0]['id']
