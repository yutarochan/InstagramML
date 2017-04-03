import numpy as np
import pandas as pd
from data import get_user_data, usernames, split_seed, test_size
from sklearn.model_selection import train_test_split

labelfilename = '../data/labels.csv'


def get_features(filename):
    # get dataframe
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    # create datasets for each user
    labels = {user: get_user_data(df, user) for user in usernames}
    # removes features that are empty in each user
    # also count number of users
    counts = {}
    for user in usernames:
        for feature in labels[user]:
            flag = labels[user][feature].isnull().all()
            if flag:
                labels[user].drop(feature, 1, inplace=True)
        # Create counts
        counts[user] = labels[user].count(axis=0)
        counts[user].drop('likes', inplace=True)
        counts[user].drop('username', inplace=True)
        counts[user].drop('display_src', inplace=True)
        # remove features with counts of less than n
        n = 10
        to_remove = counts[user][counts[user] <= n].index
        labels[user].drop(to_remove, 1, inplace=True)
        # recreate counts
        counts[user] = labels[user].count(axis=0)
        counts[user].drop('likes', inplace=True)
        counts[user].drop('username', inplace=True)
        counts[user].drop('display_src', inplace=True)
#        print counts[user].head()
#        print user, (len(labels[user].columns) - 3)
        path = '../data/labels/' + user + '.csv'
        labels[user].to_csv(path_or_buf=path, sep='\t', index=False, encoding='utf-8')
    return labels, counts


def split_data():
    """
    Split data into test and training data
    :return: 
    """
    # Get label data
    labels = {user: pd.read_csv(('../data/labels/' + user + '.csv'), sep='\t', encoding='utf-8')
              for user in usernames}
    seed = split_seed # Using this for reproducibility
    tsize = test_size # 20% of data
    train, test = {}, {}
    for user in usernames:
        # fill remaining NA with 0
        labels[user] = labels[user].fillna(0)
        data = train_test_split(labels[user], test_size=tsize, random_state=seed)
        train[user] = data[0]
        test[user] = data[1]
    return train, test


def make_csv():
    """
    Makes tab-deliminated data from the dataset. 
    Returns test and train datasets for each user.
    :return: 
    """
    train, test = split_data()
    for user in usernames:
        trainname = '../data/labels/' + 'train_' + user + '.csv'
        testname = '../data/labels/' + 'test_' + user + '.csv'
        train[user].to_csv(path_or_buf=trainname, sep='\t', index=False, encoding='utf-8')
        test[user].to_csv(path_or_buf=testname, sep='\t', index=False, encoding='utf-8')

# ** NUMBER OF LABELS **
# beautifuldestinations 522
# kissinfashion 431
# josecabaco 803
# etdieucrea 630
# instagood 731
# ** AFTER REMOVING LABELS WITH < 10 COUNT **
# beautifuldestinations 189
# kissinfashion 132
# josecabaco 181
# etdieucrea 137
# instagood 201

# how many labels correspond to the frequency (frequency: # labels)
# this probably has a pareto distribution
# 1: 383 = 1034 - 1417
# 2: 158 = 875 - 1033
# 3: 91 = 783 - 874
# 4: 64 = 718 - 782
# 5: 54 = 663 - 717
# 6: 39 = 624 - 663
# 7: 35 = 588 - 623
# 8: 24 = 563 - 587
# 9: 19 = 543 - 562
# 10: 23 = 519 - 542
# 11: 21 = 497 - 518
# 12: 23 = 473 - 496
# 13: 13 = 459 - 472
# 14: 8 = 442 - 434
# 15: 14 = 419 - 441
# 16: 8 = 411 - 418

# Frequency of number of labels has a power law distribution of A/x^b * 1/1406 with:
# A = 401.51, B = 1.312
# equivilant to alpha*x_m^(alpha)/x^(alpha+1) with:
# alpha = 0.312, x_m = 0.753

if __name__ == "__main__":
    make_csv()

