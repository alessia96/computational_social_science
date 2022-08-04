import os
import pandas as pd
import requests
import time
import random
from langdetect import detect
from datetime import datetime, date, timedelta


def date2timestamp(data):
    return int(time.mktime(datetime.strptime(data, "%Y/%m/%d").timetuple()))


def next_day(data):
    tmrw = datetime.strptime(data, "%Y/%m/%d") + timedelta(days=1)
    return tmrw.strftime('%Y/%m/%d')


def list_of_days(start_date, end_date):
    delta = end_date - start_date
    days = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        days.append(str(day).replace('-', '/'))
    return days


def scrape_reddit(subreddit, date_start, date_end, size=1000):
    """
    Parameters
    ----------
    subreddit: subreddit name
    date_start: starting date
    date_end: end date
    size: maximum number of posts per day
    Returns
    -------
    dataframe containing subreddit, author, date and post for each post in subreddit
    """
    start = date2timestamp(date_start)
    end = date2timestamp(date_end)
    # use the pushshift api to extract out data
    url = 'https://api.pushshift.io/reddit/search/submission/?subreddit={}&sort=desc&sort_type=created_utc&after={}&before={}&size={}'.format(
        subreddit, start, end, size)
    print(url)
    try:
        posts = requests.get(url)
        posts = posts.json()
        posts = posts['data']
    except:
        time.sleep(30)
        posts = requests.get(url)
        posts = posts.json()
        posts = posts['data']

    df = pd.DataFrame(columns=['subreddit', 'author', 'date', 'post'])

    for post in posts:
        if 'selftext' in post:  # check if selftext parameter exists
            text = post['selftext']
            if text != "" and text != '[removed]' and '[deleted]' not in text:  # further check if selftext is not empty
                try:
                    if detect(text) == 'en':
                        df = df.append({'subreddit': subreddit, 'author': post['author'], 'date': date_start,
                                        'post': post['title'] + ' ' + post['selftext']}, ignore_index=True)
                except:
                    continue
    return df


def save_reddit_post(subreddit_list, days, size, output_dir):
    """
    Parameters
    ----------
    subreddit_list: list of subreddits
    days: list of days (dates)
    size: maximum number of posts per day
    output_dir: directory where dataframe is saved

    Returns
    -------

    """
    # for each subreddit in the given list save a csv file containing 4 columns: subreddit, author, date, post
    for sub in subreddit_list:
        subreddit_df = pd.DataFrame(columns=['subreddit', 'author', 'date', 'post'])
        days_local = list(days)
        while subreddit_df.shape[0] < 30000 and days_local:
            idx = random.randint(0, len(days_local) - 1)
            date_start = days_local.pop(idx)
            date_end = next_day(date_start)
            df = scrape_reddit(sub, date_start, date_end, size=size)
            subreddit_df = pd.concat([subreddit_df, df])
            time.sleep(0.5)
            print(sub)
            print(subreddit_df.shape)
        subreddit_df.to_csv(os.path.join(output_dir, '{}.csv'.format(sub)), index=False)


days = list_of_days(date(2020, 1, 1), date(2022, 3, 15))
output_dir = '/PATH/Reddit/'

subreddit = ['ADHD', 'autism', 'Bipolar', 'BipolarReddit', 'Borderline', 'BorderlinePDisorder', 'CPTSD', 'OCD', 'ptsd',
             'schizoaffective', 'schizophrenia', 'anxiety', 'depression', 'EDAnonymous', 'socialanxiety',
             'suicidewatch', 'lonely', 'addiction', 'alcoholism', 'AMA', 'conspiracy', 'datascience', 'divorce',
             'fitness', 'guns', 'jokes', 'legaladvice', 'LifeProTips', 'linux', 'NoStupidQuestions', 'parenting',
             'relationships', 'teaching', 'unpopularopinion']

save_reddit_post(subreddit_list=subreddit, days=days, size=10000, output_dir=output_dir)

# create a dataset with posts of all subreddits
df = pd.DataFrame()
for sub in subreddit:
    sub_df = pd.read_csv(os.path.join(output_dir, '{}.csv'.format(sub)))
    df = df.append(sub_df)

# save full dataset
df.to_csv('/PATH/subreddit_data.csv')