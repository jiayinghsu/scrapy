# for standard data wrangling
import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
# for pattern matching during cleaning
import re
# for frequency counts
from collections import Counter
# for bigrams, conditional frequency distribution and beyond
import nltk
# for word cloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
# for (one way of) keyword extraction
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Throw out articles of unconventional formats like slideshow (resulting in NA’s)
df = pd.read_csv("analyze_news/results.csv")
print(df)

df = df.dropna(subset=['time']).reset_index(drop=True)

# format dates
# for CNN
df['date'] = df['time'].apply(lambda x: x.split('ET,')[1][4:].strip())
df.date = pd.to_datetime(df.date, format = '%B %d, %Y')
# for Fox News
# for _, row in df.iterrows():
#     if 'hour' in row['time']:
#         row['time'] = ('March 24, 2021')
#     elif 'day' in row['time']:
#         day_offset = int(row['time'].split()[0])
#         row['time'] = 'March {}, 2021'.format(24 - day_offset)
#     elif ('March' in row['time']) or ('February' in row['time']) or ('January' in row['time']):
#         row['time'] += ', 2021'
#     else:
#         row['time'] += ', 2020'
# df = df.rename(columns = {'time':'date'})
# df.date = df.date.apply(lambda x: x.strip())
# df.date = pd.to_datetime(fn.date, format = '%B %d, %Y')

df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')
df_cleaned = df[df['month_year']!=pd.Period('2020-07', 'M')].copy()

df['content'] = df['content'].apply(lambda x: x.lower())
df.content = df.content.apply(lambda x: re.sub(r'use\sstrict.*?env=prod"}', '', x))


##############
# word cloud
##############
stopwords = nltk.corpus.stopwords.words('english')
stopwords += ['the', 'says', 'say', 'a']  # add custom stopwords
stopwords_tokenized = nltk.word_tokenize(' '.join(stopwords))


def process(text):
    tokens = []
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            token = word.lower().replace("'", "")  # put words like 'she and she as one
            if ('covid-19' in token) or ('coronavirus' in token):  # publications use different terms for covid
                tokens.append('covid')  # normalize all the mentions since this is a crucial topic as of now
            else:
                tokens.append(token)
    tokens_filtered = [t for t in tokens
                       if re.search('[a-zA-Z]', t) and t not in stopwords_tokenized]
    return tokens_filtered


def gen_wc(bag, name=''):
    tokens = process(bag)
    plt.figure(figsize=(20, 10), dpi=800)
    wc = WordCloud(background_color="white", width=1000, height=500)  # other options like max_font_size=, max_words=
    wordcloud = wc.generate_from_text(' '.join(tokens))
    plt.imshow(wordcloud, interpolation="nearest", aspect="equal")
    plt.axis("off")
    plt.title('Words in Headlines-{}'.format(name))
    plt.savefig('headline_wc_{}'.format(name) + '.png', figsize=(20, 10), dpi=800)
    plt.show()


# generate word cloud for each month
for time in df['month_year'].unique():
    df_subset = df[df['month_year'] == time].copy()
    bag = df_subset['title'].str.cat(sep=' ')
    gen_wc(bag, name=time)


##############
# bigrams
##############
out = []
for title in list(df['title']):
    out.append(nltk.word_tokenize(title))
bi = []
for title_words in out:
    bi += nltk.bigrams(title_words)
Counter(bi).most_common()

cfd = nltk.ConditionalFreqDist(bi)
cfd['Covid']
# CNN: FreqDist({'relief': 8, ',': 6, 'law': 1})
cfd['coronavirus']
# Fox News: FreqDist({'pandemic': 4, 'death': 2, 'vaccine': 1, 'relief': 1, 'records': 1, 'travel': 1, 'is': 1, 'rules': 1, 'canceled': 1, ',': 1, ...})
cfd['border']
# CNN: FreqDist({'wall': 7, 'crisis': 3, 'is': 3, '.': 2, ',': 2, 'alone': 2, 'surge': 1, 'closed': 1, 'problem': 1, 'encounters': 1, ...})
# Fox News: FreqDist({'crisis': 50, 'wall': 19, ',': 14, 'surge': 13, ':': 8, 'as': 7, 'policy': 7, 'crossings': 6, "'crisis": 5, 'situation': 5, ...})

bag = df['title'].str.cat(sep = ' ')
tokens = process(bag)
word_df = pd.DataFrame.from_dict(dict(Counter(tokens)), orient='index', columns=['overall'])
# create a custom merge
def merge(original_df, frames):
    out = original_df
    for df in frames:
        out = out.merge(df, how='left', left_index=True, right_index=True)
    return out
frames = []
for time in df['month_year'].unique()[::-1]: # in reverse (chronological) order
    df_subset = df[df['month_year']==time].copy()
    bag = df_subset['title'].str.cat(sep = ' ')
    tokens = process(bag)
    frames.append(pd.DataFrame.from_dict(dict(Counter(tokens)), orient='index', columns=[str(time)]))
end_df = merge(word_df, frames)
end_df = end_df.fillna(0)

df_long_temp = end_df.drop(columns='overall').reset_index()
df_long = pd.melt(df_long_temp,id_vars=['index'],var_name='year', value_name='frequency')

# write to csv and open in tableau
df_long_temp .to_csv('df_long_temp.csv')
df_long.to_csv('df_long.csv')

##############
# key words
##############
def stemming(token):
    global stopwords_tokenized
    stemmer = SnowballStemmer("english")
    if (token in stopwords_tokenized):
        return token
    else:
        return stemmer.stem(token)


# a slightly revised process function
def preprocess(text):
    tokens = []
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            token = word.lower().replace("'", "")
            if ('covid-19' in token) or ('coronavirus' in token):
                tokens.append('covid')
            else:
                tokens.append(token)
    tokens_filtered = [t for t in tokens if re.search('[a-zA-Z]', t)]

    stems = [stemming(t) for t in tokens_filtered]
    return stems


articles = df.content.tolist()

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=200000, stop_words=stopwords_tokenized, \
                                   strip_accents='unicode', use_idf=True, tokenizer=preprocess, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(articles)

terms = tfidf_vectorizer.get_feature_names()

# pool top 10 keywords in each news article
keywords = []
for row in range(tfidf_matrix.shape[0]):
    for i in tfidf_matrix[row].toarray()[0].argsort()[::-1][:10]:
        keywords.append(terms[i])