import re
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# CLEAN, TOKENIZE AND LEMMATIZE

# load the dataset
df = pd.read_csv('/PATH/subreddit_data.csv')


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
nlp = spacy.load('en_core_web_sm')
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

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(mental_lemmas)

# Visualize the word cloud
wordcloud.to_image()

# do the same for all lemmas objects


# LDA

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('num_topics:', num_topics)
        model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=num_topics,
                         random_state=0,
                         chunksize=100,
                         alpha='auto',
                         per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('done')
        print('_______________')
    return model_list, coherence_values


# select "Borderline" subreddits
data = df.loc[df.subreddit.str.contains('orderline')]

# create a list of words
data = data.lemma.values.tolist()
data = [t.split() for t in data if type(t) != float]

# create the dictionary and the corpus
id2word = Dictionary(data)
corpus = [id2word.doc2bow(text) for text in data]

# compute best LDA model with different k (number of topics)
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data, start=2,
                                                        limit=20, step=2)

# plot a graph with the results
limit = 20
start = 2
step = 2
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
pyLDAvis.save_html(p, '/PATH/p_borderline.html')