import os
import pandas as pd
import requests
import time
import random
from langdetect import detect
from datetime import datetime, date, timedelta


class Scraper:
    def __init__(self, output_path):
        self.output_path = output_path

    def date2timestamp(self, data):
        return int(time.mktime(datetime.strptime(data, "%Y/%m/%d").timetuple()))

    def next_day(self, data):
        tmrw = datetime.strptime(data, "%Y/%m/%d") + timedelta(days=1)
        return tmrw.strftime('%Y/%m/%d')

    def list_of_days(self, start_date, end_date):
        delta = end_date - start_date
        days = []
        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i)
            days.append(str(day).replace('-', '/'))
        return days

    def scrape_reddit(self, subreddit_name, start_date, end_date, size=1000):
        """
        Parameters
        ----------
        subreddit_name: subreddit name - string
        start_date: starting date - string "%Y/%m/%d"
        end_date: end date - string "%Y/%m/%d"
        size: maximum number of posts per day
        Returns
        -------
        dataframe containing subreddit, user, date and post for each post in subreddit
        """
        start = self.date2timestamp(start_date)
        end = self.date2timestamp(end_date)
        # use the pushshift api to extract out data
        url = 'https://api.pushshift.io/reddit/search/submission/?subreddit={}&sort=desc&sort_type=created_utc&after={}&before={}&size={}'.format(
            subreddit_name, start, end, size)
        try:
            posts = requests.get(url).json()
            posts = posts['data']
        except:
            time.sleep(30)
            posts = requests.get(url).json()
            posts = posts['data']

        dataframe = pd.DataFrame(columns=['subreddit', 'user', 'date', 'post'])

        for post in posts:
            if 'selftext' in post:  # check if selftext parameter exists
                text = post['selftext']
                if text != "" and text != '[removed]' and '[deleted]' not in text:  # further check if selftext is not empty
                    try:
                        if detect(text) == 'en':
                            dataframe = dataframe.append(
                                {'subreddit': subreddit_name, 'user': post['user'], 'date': start_date,
                                 'post': post['title'] + ' ' + post['selftext']}, ignore_index=True)
                    except:
                        continue
        return dataframe

    def save_reddit_post(self, subreddit_list, start_date, end_date, size):
        """
        Parameters
        ----------
        subreddit_list: list of subreddits
        start_date: starting date - datetime.date(Y, m, d)
        end_date: end date - datetime.date(Y, m, d)
        size: maximum number of posts per day

        Returns
        -------

        """
        # for each subreddit in the given list save a csv file containing 4 columns: subreddit, user, date, post
        for subr in subreddit_list:
            last_s = None
            subreddit_df = pd.DataFrame(columns=['subreddit', 'user', 'date', 'post'])
            days_local = self.list_of_days(start_date, end_date)
            while subreddit_df.shape[0] < 30000 and days_local:
                idx = random.randint(0, len(days_local) - 1)
                date_start = days_local.pop(idx)
                date_end = self.next_day(date_start)
                dataframe = self.scrape_reddit(subr, date_start, date_end, size=size)
                subreddit_df = pd.concat([subreddit_df, dataframe])
                time.sleep(0.5)
                if subr != last_s:
                    if last_s:
                        print(f"{last_s} done \n")
                        print("_" * 25)
                    print(f"starting with {subr}")
                    last_s = subr
                print(subreddit_df.shape[0])
            subreddit_df.to_csv(os.path.join(self.output_path, '{}.csv'.format(subr)), index=False)


# create Reddit folder and Subreddits folder
os.mkdir('Reddit')
os.mkdir(os.path.join("Reddit", "Subreddit"))

output_dir = os.path.join("Reddit", "Subreddit")

subreddit = ['ADHD', 'autism', 'Bipolar', 'BipolarReddit', 'Borderline', 'BorderlinePDisorder', 'CPTSD', 'OCD', 'ptsd',
             'schizoaffective', 'schizophrenia', 'anxiety', 'depression', 'EDAnonymous', 'socialanxiety',
             'suicidewatch', 'lonely', 'addiction', 'alcoholism', 'AMA', 'conspiracy', 'datascience', 'divorce',
             'fitness', 'guns', 'jokes', 'legaladvice', 'LifeProTips', 'linux', 'NoStupidQuestions', 'parenting',
             'relationships', 'teaching', 'unpopularopinion']

scraper = Scraper(output_path=output_dir)
scraper.save_reddit_post(subreddit_list=subreddit,
                         start_date=date(2020, 1, 1),
                         end_date=date(2022, 3, 15),
                         size=10000)

# create a dataset with posts of all subreddits
df = pd.DataFrame()
for sub in subreddit:
    sub_df = pd.read_csv(os.path.join(output_dir, '{}.csv'.format(sub)))
    df = df.append(sub_df)

# save full dataset
df.to_csv(os.path.join("Reddit", "subreddit_data.csv"), index=False)
