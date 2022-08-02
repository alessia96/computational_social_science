import pandas as pd
import numpy as np
import spacy
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier, \
    LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# TFIDF

# init tfidf vectorizer ignoring terms with a frequency lower than 5%
cv = TfidfVectorizer(min_df=0.05)
tfidf = cv.fit_transform(df.lemma)

# create dataframe from tfidf
tfidf = pd.DataFrame(tfidf.todense())
tfidf.columns = cv.get_feature_names()
tfidf.insert(loc=0, column="SUBREDDIT", value=df.subreddit)


# MODELS AND PREDICTIONS
subreddit = ['ADHD', 'autism', 'Bipolar', 'BipolarReddit', 'Borderline', 'BorderlinePDisorder',
             'CPTSD', 'OCD', 'ptsd', 'schizoaffective', 'schizophrenia', 'anxiety', 'depression',
             'EDAnonymous', 'socialanxiety', 'suicidewatch', 'lonely', 'addiction', 'alcoholism']

# assign subreddits to target
# 0=mental health; 1=non-mental health
target = []
for subs in tfidf.SUBREDDIT:
    if subs in subreddit:
        target.append(0)
    else:
        target.append(1)

# noinspection PyTypeChecker
tfidf.insert(loc=0, column="GROUP", value=target)


# classification
def classification(predictors, response_var, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(predictors, response_var, test_size=test_size, random_state=42)
    train_score = []
    test_score = []
    for clf, names in ((LinearRegression(), "Linear Regression"),
                       (LogisticRegression(random_state=42), "Logistic Regression"),
                       (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                       (Perceptron(max_iter=50), "Perceptron"),
                       (SGDClassifier(max_iter=1000, tol=1e-3), "SDG Classifier"),
                       (PassiveAggressiveClassifier(max_iter=50), "PassiveAgressive Classifier"),
                       (BernoulliNB(), "Bernoulli Naive Bayes"),
                       (ComplementNB(), "Complement Naive Bayes"),
                       (MultinomialNB(), "Multinomial Naive Bayes")):
        clf.fit(X_train, y_train)
        train_score.append((names, clf.score(X_train, y_train)))
        test_score.append((names, clf.score(X_test, y_test)))
    return train_score, test_score


# plot results of classification()
def plot_models_results(train_scores, test_scores, group=0):
    colors = sns.color_palette(palette="pastel")
    groups = {0: '"mental health" and "non-mental health"',
              1: '"Borderline" and "Bipolar"'}
    fig, ax = plt.subplots(figsize=(15, 8), dpi=80)
    x = np.arange(len(train_scores))
    # barplots
    l_bar = ax.bar(x - 0.35 / 2, [i[1] for i in train_scores], width=0.35, label='Train Score',
                   color=colors[2],
                   edgecolor='black',
                   zorder=3,
                   linewidth=2)
    r_bar = ax.bar(x + 0.35 / 2, [i[1] for i in test_scores], width=0.35, label='Test Score',
                   color=colors[0],
                   edgecolor='black',
                   zorder=3,
                   linewidth=2)
    ax.grid(zorder=0, axis='y', which='major', alpha=0.6, linewidth=0.4)    # horizontal lines
    # add xtick labels
    labels = ['\n'.join(i[0].split()) for i in train_scores]
    ax.set_xticks(range(len(train_scores)))
    ax.set_xticklabels(labels)
    # remove borders
    for i in ['top', 'bottom', 'right', 'left']:
        ax.spines[i].set_visible(False)
    # remove tick marks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    # legend and title
    fig.legend(loc='upper center', ncol=2, frameon=False)
    ax.set_title(f'Train and test results {groups[group]} subreddits classification', fontsize=18)
    plt.show()


# get train and test scores for all classification methods
train_score, test_score = classification(tfidf[tfidf.columns[2:]], tfidf.GROUP, test_size=0.2)
plot_models_results(train_score, test_score, group=0)


# select bipolar and borderline subreddits
b = tfidf.loc[tfidf.SUBREDDIT.str.contains('orderline')]
b = b.append(tfidf.loc[tfidf.SUBREDDIT.str.contains('ipolar')])

# group = 0 if borderline, group = 1 if bipolar
b['GROUP'] = [0 if 'orderline' in i else 1 for i in b.SUBREDDIT]

# compute train and test scores for borderline and bipolar
train_score2, test_score2 = classification(b[b.columns[2:]], b.GROUP, test_size=0.2)

# create dataframe containing all scores for both groups
stats = pd.DataFrame({'method': [i[0] for i in train_score],
                      'full_train': [i[1] for i in train_score],
                      'full_test': [i[1] for i in test_score],
                      'b_train': [i[1] for i in train_score2],
                      'b_test': [i[1] for i in test_score2]})

print(stats)

