import os
import re
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

nltk.download('averaged_perceptron_tagger')


class Cleaner:

    def __init__(self, data):
        """
        Parameters
        ----------
        data: list of string or series

        Returns
        -------

        """

        self.data = data
        self.lemmatizer = WordNetLemmatizer()

    # clean text
    def clean_str(self, text):
        """
        Parameters
        ----------
        text: string to clean
        Returns
        -------
        cleaned string
        """
        # all lowercase
        text = text.lower()
        # remove stopwords
        text = " ".join([word for word in text.split() if word not in set(stopwords.words("english"))])
        # remove urls
        text = re.sub("https?:\/\/.*[\r\n]*", "", text)
        # remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # remove numbers
        text = re.sub(r'\d+', '', text)
        # remove white spaces and single strings
        alph = set([chr(x) for x in range(ord('a'), ord('z') + 1)])
        text = ' '.join(text.split())
        text = ' '.join([w for w in text.split() if w not in alph])
        return text

    def pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, clean=True, tokenize=True, lemmatize=True):
        out = pd.DataFrame({"original": self.data})
        if clean:
            cleaned = self.data.apply(lambda row: self.clean_str(row))
            out['clean'] = cleaned
        else:
            cleaned = self.data
        if tokenize:
            token = cleaned.apply(lambda row: nltk.word_tokenize(row))
            out['token'] = token
        else:
            token = self.data.apply(lambda row: row.split())
        if lemmatize:
            pos_tagged = token.apply(lambda row: nltk.pos_tag(row))
            wordnet_tagged = pos_tagged.apply(lambda row: list(map(lambda w: (w[0], self.pos_tagger(w[1])), row)))
            lemmas = wordnet_tagged.apply(
                lambda lst: [self.lemmatizer.lemmatize(word[0], pos=word[1]) if word[1] else word[0] for word in lst])
            out['lemma'] = [' '.join(i) for i in lemmas]
        return out


class LDA:

    def __init__(self, data):
        self.data = data

    def to_lda(self):
        texts = list(self.data.apply(lambda row: row.split()))
        id2word = Dictionary(texts)
        corpus_bow = [id2word.doc2bow(text) for text in texts]
        return texts, id2word, corpus_bow

    def compute_coherence_values(self, id2word=None, corpus_bow=None, texts=None, min_topic=2, max_topic=10, steps=1):
        """
        Parameters
        ----------
        id2word: dictionary created from texts
        corpus_bow: bag of words created from texts and dictionary
        texts: list of strings
        min_topic: minimum number of topics - default=2
        max_topic: maximum number of topics - default=2
        steps: step between number of topics

        Returns
        -------
        list of lda models
        list of coherence values for each model
        """

        if not texts or not id2word or not corpus_bow:
            texts, id2word, corpus_bow = self.to_lda()

        coherence_values_lst = []
        model_lst = []
        for num_topics in range(min_topic, max_topic, steps):
            print(f'number of topics: {num_topics}')
            model = LdaModel(corpus=corpus_bow,
                             id2word=id2word,
                             num_topics=num_topics,
                             random_state=42,
                             chunksize=100,
                             alpha='auto',
                             per_word_topics=True)
            model_lst.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
            coherence_values_lst.append(coherencemodel.get_coherence())
            print('done')
            print('_______________\n')
        return model_lst, coherence_values_lst

    def best_lda(self, id2word=None, corpus_bow=None, texts=None, min_topic=2, max_topic=10, steps=1):
        if not texts or not id2word or not corpus_bow:
            texts, id2word, corpus_bow = self.to_lda()
        model_lst, coherence_values_lst = self.compute_coherence_values(id2word, corpus_bow, texts, min_topic,
                                                                        max_topic, steps)
        best_result_idx = coherence_values_lst.index(max(coherence_values_lst))
        best_model = model_lst[best_result_idx]
        return best_model


# ANALYSIS

# plot top-n words
def plot_top_words(lemmas, quantity=10, color='blue'):
    """
    Parameters
    ----------
    lemmas: unique string of lemmas
    quantity: number of top words to plot - default=10
    color: color of bars - default=blue
    Returns
    -------
    horizontal bar-plot with top-n words
    """
    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(lemmas)
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
def plot_wordcloud(lemmas):
    """
    Parameters
    ----------
    lemmas: unique string of lemmas

    Returns
    -------
    plot wordcloud
    """
    # create wordcloud object
    wordcloud = WordCloud(background_color="black", max_words=5000, contour_width=3, contour_color='steelblue')
    # generate wordcloud
    wordcloud.generate(lemmas)
    # Visualize wordcloud
    wordcloud.to_image().show()


# load the dataset
df = pd.read_csv(os.path.join("Reddit", "subreddit_data.csv"))
df = df.loc[~df.user.str.contains('AutoModerator')]
df = df.dropna()

# Cleaning and lemmatization

# lemmatize posts
cleaner = Cleaner(df.post)
clean_df = cleaner.lemmatize()

clean_df.rename(columns={"original": "post"}, inplace=True)
df = pd.merge(df, clean_df, on="post", how="inner")

# save cleaned df
df.to_csv(os.path.join("Reddit", "subreddit_cleaned.csv"))


# Analysis
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


# example plot top 25 words and wordcloud for borderline subreddits
plot_top_words(borderline_lemmas, 25, 'violet')
plot_wordcloud(borderline_lemmas)


# example plot top 25 words and wordcloud for borderline subreddits
plot_top_words(borderline_lemmas, 25, 'violet')
plot_wordcloud(borderline_lemmas)


# example LDA with Borderline subreddits

bth = set(border.user.unique()).intersection(set(bipolar.user.unique()))
border = border.loc[~border.user.isin(bth)]

lda = LDA(border.lemma)

# get best lda model
optimal_model = lda.best_lda(max_topic=20)

# save the model
lda_folder = os.path.join("Reddit", "LDA models")
os.mkdir(lda_folder)
optimal_model.save(os.path.join(lda_folder, "lda_model_borderline.model"))


# produce a pyLDAvis visualization of the optimal model and save it
data, dictionary, corpus = lda.to_lda()
p = gensimvis.prepare(optimal_model, corpus, dictionary)
pyLDAvis.save_html(p, os.path.join(lda_folder, "borderline_LDA.html"))
