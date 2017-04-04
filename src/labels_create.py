'''
InstagramML - Annotations Preprocessing
Author: Bill Dusch (bill.dusch@gmail.com)
'''

import numpy as np
import pandas as pd
import data
from random import randint
import operator

def get_annotations():
    filename = data.filename
    print "Parsing JSON data..."
    raw_data = data.parse_json(filename)
    num_accounts = len(raw_data)
    print "Reading DataFrame..."
    df = pd.read_csv('../data/alldata.csv', sep='\t', encoding='utf-8')
    size = len(df)
    # indices for dataframe
    print "Preparing Label DataFrame..."
    index = [x for x in range(size)]
    columns = ['likes', 'username', 'display_src', 'hour', 'day']
    tagdf = pd.DataFrame(index=index, columns=columns)
    x = 0

    tagdf['likes'] = df['likes']
    tagdf['username'] = df['username']
    tagdf['display_src'] = df['display_src']
    tagdf['hour'] = pd.to_datetime(df['date'], unit='s').dt.hour
    tagdf['day'] = pd.to_datetime(df['date'], unit='s').dt.weekday
    print "Starting Iteration"
    for i in range(num_accounts):
        posts = raw_data[i]['posts']
        for j in range(len(posts)):
            # annotations
            annotations = posts[j]['annotations']
            # 1416 labels with score: 0 - 1
            if 'labelAnnotations' in annotations:
                # Sets each label as a new feature with its  value the label's score
                for item in annotations['labelAnnotations']:
                    description = item['description'].encode('utf-8').replace(' ', '_').replace("'", "")
                    score = np.float(item['score'])
                    # This also sets new columns to NaN
                    tagdf.set_value(x, description, score)
#            if 'faceAnnotations' in annotations:
#                pass
            # extracts texts from image... do we need this?
#            if 'textAnnotations' in annotations:
#                pass
#            if 'webDetection' in annotations:
#                pass
#            if 'imagePropertiesAnnotation' in annotations:
#                pass
#            if 'fullTextAnnotation' in annotations:
#                pass
#            if 'cropHintsAnnotation' in annotations:
#                pass
            # This format is simple: 4 features, rating
#            if 'safeSearchAnnotation' in annotations:
#                violence = str(annotations['safeSearchAnnotation']['violence'])
#                print(likes, violence)
                # VERY_UNLIKELY, UNLIKELY
                # medical
                # spoof
                # violence
                # adult
#                pass
            print x
            x += 1
    return tagdf

# raw_data = data.parse_json(data.filename)
# i, j = randint(0, 4), randint(0, 900)
# anno = raw_data[i]['posts'][j]['annotations']
# print(raw_data[i]['posts'][j]['instagram']['display_src'])
# print(i, j)
# print(anno['webDetection'])
#for item in anno['labelAnnotations']:
#    print('{}: {}'.format(item['description'], item['score']))

if __name__ == '__main__':
    tagdf = get_annotations()
    print(tagdf.head())
    tagdf.to_csv(path_or_buf='../data/labels.csv', sep='\t', index=False, encoding='utf-8')
