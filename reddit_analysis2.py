import os
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats


# TFIDF
def tfidf_matrix(lemmas_lst):
    """
    Parameters
    ----------
    lemmas_lst: list or series of lemmas

    Returns
    -------
    pandas dataframe
    """
    # init tfidf vectorizer ignoring terms with a frequency lower than 5% and higher than 95%
    cv = TfidfVectorizer(min_df=0.05, max_df=0.95, stop_words='english')
    tfidf_matx = cv.fit_transform(lemmas_lst)

    # create dataframe from tfidf
    tfidf_matx = pd.DataFrame(tfidf_matx.todense())
    tfidf_matx.columns = cv.get_feature_names()
    return tfidf_matx


df = pd.read_csv(os.path.join("Reddit", "subreddit_cleaned.csv"))

# create TF-IDF matrix from lemmatized posts
tfidf = tfidf_matrix(df.lemma)
tfidf.insert(loc=0, column="SUBREDDIT", value=list(df.subreddit))
tfidf.insert(loc=0, column="USER", value=list(df.user))

# MODELS AND PREDICTIONS

subreddit = ['ADHD', 'autism', 'Bipolar', 'BipolarReddit', 'Borderline', 'BorderlinePDisorder',
             'CPTSD', 'OCD', 'ptsd', 'schizoaffective', 'schizophrenia', 'anxiety', 'depression',
             'EDAnonymous', 'socialanxiety', 'suicidewatch', 'lonely', 'addiction', 'alcoholism']

# assign subreddits to target
# 0=mental health; 1=non-mental health
target = [0 if subr in subreddit else 1 for subr in tfidf.SUBREDDIT]

# noinspection PyTypeChecker
tfidf.insert(loc=0, column="Y", value=target)


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
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, response_var, test_size=test_size, random_state=42)
    train_scores = []
    test_scores = []
    for clf, names in ((LogisticRegression(random_state=42), "Logistic Regression"),
                       (RidgeClassifier(tol=1e-2, solver="sag", random_state=42), "Ridge Classifier"),
                       (Perceptron(max_iter=50), "Perceptron"),
                       (SGDClassifier(max_iter=1000, tol=1e-3, random_state=42), "SDG Classifier"),
                       (PassiveAggressiveClassifier(max_iter=50), "PassiveAgressive Classifier"),
                       (BernoulliNB(), "Bernoulli Naive Bayes"),
                       (ComplementNB(), "Complement Naive Bayes"),
                       (MultinomialNB(), "Multinomial Naive Bayes"),
                       (KNeighborsClassifier(n_neighbors=11), "KNN")
                       ):
        clf.fit(X_train, Y_train)
        train_scores.append((names, clf.score(X_train, Y_train)))
        test_scores.append((names, clf.score(X_test, Y_test)))
    return train_scores, test_scores


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
    groups = {0: '"mental health" and "non-mental health"',
              1: '"Borderline" and "Bipolar"'}
    fig, ax = plt.subplots(figsize=(15, 8), dpi=80)
    x = np.arange(len(train_scores))

    # barplots
    ax.bar(x - 0.35 / 2, [i[1] for i in train_scores], width=0.35, label='Train Score',
           color='lime', edgecolor='black', zorder=3, linewidth=2)
    ax.bar(x + 0.35 / 2, [i[1] for i in test_scores], width=0.35, label='Test Score',
           color='cyan', edgecolor='black', zorder=3, linewidth=2)

    ax.grid(zorder=0, axis='y', which='major', alpha=0.6, linewidth=0.4)  # horizontal lines

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
train_score, test_score = classification(tfidf[tfidf.columns[3:]], tfidf.Y, test_size=0.2)
#plot_models_results(train_score, test_score, group=0)

# repeat all for Bipolar and Borderline
# select bipolar and borderline subreddits
btfidf = tfidf.loc[tfidf.SUBREDDIT.str.contains('orderline')]
btfidf = btfidf.append(tfidf.loc[tfidf.SUBREDDIT.str.contains('ipolar')])

btarget = [0 if "orderline" in subr else 1 for subr in btfidf.SUBREDDIT]

# noinspection PyTypeChecker
btfidf["Y"] = btarget

train_score2, test_score2 = classification(btfidf[btfidf.columns[3:]], btfidf.Y, test_size=0.2)

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


def plot_conf_matrix(Y_test, Y_pred):
    """
    Parameters
    ----------
    Y_test: true Y
    Y_pred: predicted Y (y^)

    Returns
    -------
    confusion matrix plot
    """
    cm = confusion_matrix(Y_test, Y_pred)
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    sns.heatmap(df_cm, annot=True, fmt='.0f')
    plt.show()


def plot_roc(Y_test, Y_pred):
    """
    Parameters
    ----------
    Y_test: true Y
    Y_pred: predicted Y (y^)

    Returns
    -------
    roc-curve plot
    """
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def top_estimates(model, X, y, features):
    """
    Parameters
    ----------
    model: fitted model e.g. LogisticRegression
    X: predictors used to fit the model
    y: response variable used to fit the model
    features: feature names

    Returns
    -------
    g1: top 25 features group1 with summary
    g2: top 25 features group2 with summary
    """
    model_summ = summary(model, X, y, features)
    model_summ = model_summ.loc[model_summ['p-value'] <= 0.05]
    g1 = model_summ.loc[model_summ.estimate < 0].sort_values(by="estimate", ascending=False).tail(25)
    g2 = model_summ.loc[model_summ.estimate > 0].sort_values(by="estimate").tail(25)
    g1['estimate'] = g1.estimate * -1
    return g1.iloc[::-1], g2.iloc[::-1]


def plot_estimates(model, X, y, features):
    """
    Parameters
    ----------
    model: fitted model e.g. LogisticRegression
    X: predictors used to fit the model
    y: response variable used to fit the model
    features: feature names

    Returns
    -------
    bar-plot of 25 features per group
    """
    g1, g2 = top_estimates(model, X, y, features)
    g1 = g1.iloc[::-1]
    g2 = g2.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), dpi=80)
    x1 = np.arange(len(g1))
    x2 = np.arange(len(g2))
    ax[0].barh(x1, g1.estimate, color='cyan', edgecolor='black', zorder=3, linewidth=2, label='Group-1')
    ax[1].barh(x2, g2.estimate, color='lime', edgecolor='black', zorder=3, linewidth=2, label='Group-2')
    for i in range(2):
        ax[i].set_xticks(np.arange(max(max(g1.estimate), max(g2.estimate)) + 1))
        ax[i].set_yticks(x1) if i == 0 else ax[i].set_yticks(x2)
        ax[i].set_yticklabels(g1.index) if i == 0 else ax[i].set_yticklabels(g2.index)
        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')
        for j in ['top', 'bottom', 'right', 'left']:
            ax[i].spines[j].set_visible(False)
        ax[i].grid(zorder=0, axis='x', which='both', alpha=0.6, linewidth=0.4)
        ax[i].set_title(f'25 most important features for Group-{i + 1}')
    plt.show()


feat_names = tfidf.columns[3:]
x_train, x_test, y_train, y_test = train_test_split(tfidf[feat_names], tfidf.Y, test_size=0.2, random_state=42)
LR = LogisticRegression(random_state=42)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)

plot_conf_matrix(y_test, y_pred)
plot_roc(y_test, y_pred)
print(classification_report(y_test, y_pred))

model_summary = summary(LR, x_train, y_train, feat_names)
print(model_summary)

plot_estimates(LR, x_train, y_train, feat_names)

estimates1, estimates2 = top_estimates(LR, x_train, y_train, feat_names)
