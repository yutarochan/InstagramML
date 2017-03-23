'''
InstagramML - Data Parser & Preprocessing
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import json
import pandas as pd
import numpy as np

def parse_json(filename):
    json_file = open(filename, 'rb')
    return json.loads(json_file.readlines()[0])

def download_images(target_folder, json):
    parse_json

def dataframe(filename):
    raw_data = parse_json(filename)
    num_accounts = len(raw_data)
    columns = ['likes', 'username', 'id', 'date', 'instagram_id', 'thumbnail_src', 'display_src', 'video',
               'height', 'width', 'caption']
    # get total number of files
    size = 0
    for i in range(num_accounts):
        size += len(raw_data[i]['posts'])
    index = [x for x in range(size)]
    df = pd.DataFrame(index=index, columns=columns)
    x = 0
    for i in range(num_accounts):
        username = str(raw_data[i]['username'])
        id = int(raw_data[i]['id'])
        posts = raw_data[0]['posts']
        for j in range(len(posts)):
            # Add posts to updated
            df['likes'][x] = posts[j]['instagram']['likes']['count']
            df['username'][x] = username
            df['id'][x] = id
            df['date'][x] = posts[j]['instagram']['date']
            df['instagram_id'][x] = posts[j]['instagram']['id']
            df['thumbnail_src'][x] = posts[j]['instagram']['thumbnail_src']
            df['display_src'][x] = posts[j]['instagram']['display_src']
            df['video'][x] = posts[j]['instagram']['is_video']
            df['height'][x] = posts[j]['instagram']['dimensions']['height']
            df['width'][x] = posts[j]['instagram']['dimensions']['width']
            df['caption'][x] = posts[j]['instagram']['caption']
            annotations = posts[j]['annotations']
            # ** insert annotation code here... **
            # faceAnnotations
            if 'faceAnnotations' in annotations:
                pass
            if 'labelAnnotations' in annotations:
                pass
            if 'textAnnotations' in annotations:
                pass
            if 'webDetection' in annotations:
                pass
            if 'imagePropertiesAnnotation' in annotations:
                pass
            if 'fullTextAnnotation' in annotations:
                pass
            if 'cropHintsAnnotation' in annotations:
                pass
            if 'safeSearchAnnotation' in annotations:
                pass
            # count for dataframe
            x += 1
    return df


filename = '../data/dataset.json'

# Test
raw_data = parse_json(filename)

df = dataframe(filename)
print(df.head())
