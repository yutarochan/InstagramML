'''
InstagramML - Data Parser & Preprocessing
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)

Functions
---------
parse_json
download_images
dataframe
split_data(filename)
    Take the data directly from the filename and return a split test and training dataset
'''
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# * * * GLOBAL VARIABLES * * *
usernames = ['kissinfashion', 'instagood', 'beautifuldestinations', 'etdieucrea', 'josecabaco']
# For scoring
thresholds = {'kissinfashion': 0.095, 'instagood': 0.145, 'beautifuldestinations': 0.105, 'etdieucrea': 0.145, 'josecabaco': 0.185}
filename = '../data/dataset.json'
trainname = '../data/train.csv'
testname = '../data/test.csv'


def parse_json(filename):
    json_file = open(filename, 'rb')
    return json.loads(json_file.readlines()[0])


def download_images(target_folder, json):
#    parse_json
    pass


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
        posts = raw_data[i]['posts']
        for j in range(len(posts)):
            # Add posts to updated
            df.set_value(x, 'likes', posts[j]['instagram']['likes']['count'])
            df.set_value(x,'username', username)
            df.set_value(x, 'id', id)
            df.set_value(x, 'date', posts[j]['instagram']['date'])
            df.set_value(x, 'instagram_id', posts[j]['instagram']['id'])
            df.set_value(x, 'thumbnail_src', posts[j]['instagram']['thumbnail_src'])
            df.set_value(x, 'display_src', posts[j]['instagram']['display_src'])
            df.set_value(x, 'video', posts[j]['instagram']['is_video'])
            df.set_value(x, 'height', posts[j]['instagram']['dimensions']['height'])
            df.set_value(x, 'width', posts[j]['instagram']['dimensions']['width'])
            if 'caption' in posts[j]['instagram']:
                df.set_value(x, 'caption', posts[j]['instagram']['caption'])
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


def get_user_data(df, username):
    """
    From the whole data, return the dataframe corresponding to one username.
    :param df: Dataframe of all the data
    :param username: username
    :return: 
    """
    return df.loc[df['username'] == username]


def split_data(filename):
    # Get full dataframe
    df = dataframe(filename)
    users = {user: get_user_data(df, user) for user in usernames}
    seed = 3075 # Using this for reproducibility
    test_size = 0.20 # 29% of data
    train, test = {}, {}
    for user in users:
        data = train_test_split(users[user], test_size=test_size, random_state=seed)
        train[user] = data[0]
        test[user] = data[1]
    return train, test


def make_csv(filename):
    """
    Makes tab-deliminated data from the dataset. 
    Returns test and train datasets for each user.
    :return: 
    """
    usernames = ['kissinfashion', 'instagood', 'beautifuldestinations', 'etdieucrea', 'josecabaco']
    train, test = split_data(filename)
    for user in usernames:
        trainname = '../data/' + 'train_' + user + '.csv'
        testname = '../data/' + 'test_' + user + '.csv'
        train[user].to_csv(path_or_buf=trainname, sep='\t', index=False, encoding='utf-8')
        test[user].to_csv(path_or_buf=testname, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    # Test
#    raw_data = parse_json(filename)

#    train, test = split_data(filename)
#    print(train.head())
#    print(test.head())
#    print(len(train), len(test))
#    df = dataframe(filename)
#    df.to_csv(path_or_buf='../data/alldata.csv', sep='\t', index=False, encoding='utf-8')
    make_csv(filename)


