import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier, \
    LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats

df = pd.read_csv('/PATH/subreddit_cleaned.csv')

# TFIDF

# init tfidf vectorizer ignoring terms with a frequency lower than 5%
cv = TfidfVectorizer(min_df=0.05, max_df=0.95, stop_words='english')
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
    """
    Parameters
    ----------
    predictors: dataframe of predictors
    response_var: response variable - list or series
    test_size: size of the test_set - default=0.2

    Returns
    -------
    list of train scores: performance (accuracy) of models on train set
    list of test scores: performance (accuracy) of models on test set
    """
    X_train, X_test, y_train, y_test = train_test_split(predictors, response_var, test_size=test_size, random_state=42)
    train_score = []
    test_score = []
    for clf, names in ((LogisticRegression(random_state=42), "Logistic Regression"),
                       (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                       (Perceptron(max_iter=50), "Perceptron"),
                       (SGDClassifier(max_iter=1000, tol=1e-3), "SDG Classifier"),
                       (PassiveAggressiveClassifier(max_iter=50), "PassiveAgressive Classifier"),
                       (BernoulliNB(), "Bernoulli Naive Bayes"),
                       (ComplementNB(), "Complement Naive Bayes"),
                       (MultinomialNB(), "Multinomial Naive Bayes"),
                       (KNeighborsClassifier(n_neighbors=11), "KNN"),
                       (RandomForestClassifier(random_state=42), "Random Forest")
                       ):
        clf.fit(X_train, y_train)
        train_score.append((names, clf.score(X_train, y_train)))
        test_score.append((names, clf.score(X_test, y_test)))
    return train_score, test_score


# plot results of classification()
def plot_models_results(train_scores, test_scores, group=0):
    """
    Parameters
    ----------
    train_scores: list of performances of models on train set
    test_scores: list of performances of models on test set
    group: 0 = "mental vs non-mental", 1 = "borderline vs bipolar" - default=0
    Returns
    -------
    bar-plot of performances for each model and group
    """
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


# repeat all for Bipolar and Borderline
# select bipolar and borderline subreddits
btfidf = tfidf.loc[tfidf.SUBREDDIT.str.contains('orderline')]
btfidf = btfidf.append(tfidf.loc[tfidf.SUBREDDIT.str.contains('ipolar')])

btarget = []
for subs in btfidf.subreddit:
    if 'orderline' in subs:
        btarget.append(0)
    else:
        btarget.append(1)

# noinspection PyTypeChecker
btfidf["GROUP"] = btarget

train_score2, test_score2 = classification(btfidf[btfidf.columns[1:]], btfidf.GROUP, test_size=0.2)


# create dataframe containing all scores for both groups
performances = pd.DataFrame({'method': [i[0] for i in train_score],
                      'full_train': [i[1] for i in train_score],
                      'full_test': [i[1] for i in test_score],
                      'b_train': [i[1] for i in train_score2],
                      'b_test': [i[1] for i in test_score2]})

print(performances)

# returns summary as R
def summary(model, X, y, features):
    """
    Parameters
    ----------
    model: fitted model e.g. LogisticRegression
    X: predictors used to fit the model
    y: response variable used to fit the model
    features: feature names

    Returns
    -------
    summary of fitted model (as R): for each feature the estimate, standard error,
    t-statistic, p-value, and significance
    """
    sse = np.sum((model.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
    t_stat = model.coef_ / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), y.shape[0] - X.shape[1]))
    p_val = np.round(p_val, 7)
    sig = ['*' if p <= 0.05 else "" for p in p_val[0]]
    res = pd.DataFrame({"estimate": model.coef_[0],
                  "std. error": se[0],
                  "t-statistic": t_stat[0],
                  "p-value": p_val[0],
                  "": sig})
    res.index = features
    return res


def plot_conf_matrix(y_test, y_pred):
    """
    Parameters
    ----------
    y_test: true Y
    y_pred: predicted Y (y^)

    Returns
    -------
    confusion matrix plot
    """
    cm = confusion_matrix(y_test, y_pred)
    # Create a dataframe with the confussion matrix values
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    sns.heatmap(df_cm, annot=True, fmt='.0f')
    plt.show()


def plot_roc(y_test, y_pred):
    """
    Parameters
    ----------
    y_test: true Y
    y_pred: predicted Y (y^)

    Returns
    -------
    roc-curve plot
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


feat_names = tfidf.columns[2:]
X_train, X_test, y_train, y_test = train_test_split(tfidf[feat_names], tfidf.GROUP, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


plot_conf_matrix(y_test, y_pred)
plot_roc(y_test, y_pred)
print(classification_report(y_test, y_pred))

model_summary = summary(clf, X_train, y_train, feat_names)
print(model_summary)

