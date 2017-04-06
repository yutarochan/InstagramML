import numpy as np
import pandas as pd
import seaborn as sns
from data import usernames, thresholdsfor user in usernames:
    sns.distplot(labels[user]['likes'])

# labelname = '../data/labels/' + user + '.csv'
# trainname = '../data/labels/' + 'train_' + user + '.csv'
# testname = '../data/labels/' + 'test_' + user + '.csv'


labels = {user: pd.read_csv(('../data/labels/' + user + '.csv'), sep='\t', encoding='utf-8')
              for user in usernames}
train = {user: pd.read_csv(('../data/labels/' + 'train_' + user + '.csv'), sep='\t', encoding='utf-8')
              for user in usernames}
test = {user: pd.read_csv(('../data/labels/' + 'test_' + user + '.csv'), sep='\t', encoding='utf-8')
              for user in usernames}

