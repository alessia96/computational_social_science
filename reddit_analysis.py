import re
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier, \
    LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# CLEAN, TOKENIZE AND LEMMATIZE

# load the dataset
df = pd.read_csv('/PATH/subreddit_data.csv')


# clean text
def clean(text):
    # all lowercase
    text = text.lower()
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    # remove urls
    text = re.sub("https?:\/\/.*[\r\n]*", "", text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove white spaces
    text = ' '.join(text.split())
    return text


stop_words = set(stopwords.words("english"))
# nlp = spacy.load('en_core_web_sm')
wordnet_lemmatizer = WordNetLemmatizer()

df['clean'] = df.apply(lambda row: clean(row['post']), axis=1)
df['token'] = df.apply(lambda row: nltk.word_tokenize(row['clean']), axis=1)
df['lemma'] = df.token.apply(lambda lst: [wordnet_lemmatizer.lemmatize(word, pos='v') for word in lst])
df.lemma = [' '.join(i) for i in df.lemma]


# ANALYSIS

subreddit = ['ADHD', 'autism', 'Bipolar', 'BipolarReddit', 'Borderline', 'BorderlinePDisorder',
             'CPTSD', 'OCD', 'ptsd', 'schizoaffective', 'schizophrenia', 'anxiety', 'depression',
             'EDAnonymous', 'socialanxiety', 'suicidewatch', 'lonely', 'addiction', 'alcoholism']

# separate mental health and non-mental health subreddits
mental = df.loc[df.subreddit.isin(subreddit)]
non_mental = df.loc[~df.subreddit.isin(subreddit)]

# separate Borderline and Bipolar subreddits
border = df.loc[df.subreddit.str.contains('orderline')]
bipolar = df.loc[df.subreddit.str.contains('ipolar')]

# Join the lemmas.
mental_lemmas = ','.join(list(mental['lemma'].values))
non_mental_lemmas = ','.join(list(non_mental['lemma'].values))
borderline_lemmas = ','.join(list(border['lemma'].values))
bipolar_lemmas = ','.join(list(bipolar['lemma'].values))


# plot top-n words
def plot_top_words(lemma_lst, quantity=10, color='blue'):
    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(lemma_lst)
    freq = nltk.FreqDist(tokens)
    freq = pd.DataFrame({"Word": list(freq.keys()), "Frequency": list(freq.values())})

    # reverse the order -- bigger on the top
    freq = freq.nlargest(columns="Frequency", n=quantity)[::-1]

    fig, ax = plt.subplots(figsize=(8, 10), dpi=80)
    # change plot background color
    ax.set_facecolor('lightgrey')

    # horizontal bar plot
    ax.barh(freq['Word'], freq['Frequency'],
            height=0.8,
            color=color,
            edgecolor='black',
            zorder=3)
    # set x, y ticks
    labels = [i for i in freq.Word]
    ax.set_yticks(range(quantity))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(0, max(freq.Frequency) + 5000, 5000))
    ax.set_xlim(0, max(freq.Frequency) + 5000)

    # horizontal lines
    ax.grid(zorder=0, axis='both', which='major',
            alpha=0.6, linewidth=0.4)

    # remove borders
    for i in ['top', 'bottom', 'right', 'left']:
        ax.spines[i].set_visible(False)

    # remove tick marks
    ax.yaxis.set_ticks_position('none')

    ax.set_title(f'Top {quantity} Words', fontsize=18)
    plt.show()


# plot wordcloud
def plot_wordcloud(lemma_lst):
    # create wordcloud object
    wordcloud = WordCloud(background_color="black", max_words=5000, contour_width=3, contour_color='steelblue')
    # generate wordcloud
    wordcloud.generate(lemma_lst)
    # Visualize wordcloud
    wordcloud.to_image().show()


# example plot top 25 words and wordcloud for borderline subreddits

plot_top_words(borderline_lemmas, 25, 'violet')
plot_wordcloud(borderline_lemmas)


# LDA

# Compute c_v coherence for various number of topics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(f'number of topics: {num_topics}')
        model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=num_topics,
                         random_state=42,
                         chunksize=100,
                         alpha='auto',
                         per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('done')
        print('_______________')
    return model_list, coherence_values


# example with Borderline subreddits
# select "Borderline" subreddits
data = df.loc[df.subreddit.str.contains('orderline')]

# create a list of words
data = data.lemma.values.tolist()
data = [t.split() for t in data if type(t) != float]

# create the dictionary and the corpus
id2word = Dictionary(data)
corpus = [id2word.doc2bow(text) for text in data]

# compute best LDA model with different k (number of topics)
limit = 20
start = 2
step = 1
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data, start=start,
                                                        limit=limit, step=step)

# plot a graph with the results
x = range(start, limit, step)
plt.plot(x, coherence_values, ls='-', marker='o')
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# find the best model (higher coherence value)
best_result_index = coherence_values.index(max(coherence_values))
optimal_model = model_list[best_result_index]

# produce a pyLDAvis visualization of the optimal model and save it
p = gensimvis.prepare(optimal_model, corpus, id2word)
pyLDAvis.save_html(p, '/home/a/p_borderline.html')

# save the model
optimal_model.save('/PATH/lda_model.model')

# TFIDF

# init tfidf vectorizer ignoring terms with a frequency lower than 5%
cv = TfidfVectorizer(min_df=0.05)
tfidf = cv.fit_transform(df.lemma)

# create dataframe from tfidf
tfidf = pd.DataFrame(tfidf.todense())
tfidf.columns = cv.get_feature_names()
tfidf.insert(loc=0, column="SUBREDDIT", value=df.subreddit)


# MODELS AND PREDICTIONS

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

