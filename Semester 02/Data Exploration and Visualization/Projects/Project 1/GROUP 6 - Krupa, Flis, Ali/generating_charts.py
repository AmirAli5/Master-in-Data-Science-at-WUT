import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from matplotlib.pyplot import gca
from PIL import Image
from dateutil.parser import parse
import re


data = pd.read_csv("./data/trump_tweets.csv")


trump_head = np.array(Image.open("head.png"))
stopwords = set(STOPWORDS) | {'t', 'co', 'https', 'realDonaldTrump', 'RT', 'amp', 's', 'u', 'will', 'M', 'Trump'}

wordcloud = WordCloud(width = 6000,max_words=98, height = 4000, random_state=1, background_color='#013777',
                      colormap='Pastel1', collocations=False, stopwords = stopwords, mask=trump_head)
wordcloud.generate(' '.join(data['text']))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout()
plt.savefig('head.png')

def clean_text(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", text).split())

sentiment = []
for index, tweet in data.iterrows():
    sentiment.append(TextBlob(clean_text(tweet.text)).sentiment.polarity)

sentiment = np.array(sentiment)

data['datetime'] = data.apply(lambda row: parse(row.date), axis=1)
data = data.drop(columns=['date'])

data['sentiment'] = 0
data['sentiment'] = data['sentiment']+(sentiment>0)*1+(sentiment<0)*(-1)
data['year'] = data.datetime.dt.year
sen = data.groupby(['sentiment']).count()
print(sen)

# idx = sen.index.get_level_values(0)

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 14,
        }
sen = sen.values[:,0]
pie, ax = plt.subplots(figsize=[10,6])
colors = ['#c81f2d', '#ece2a7', '#26d0ff']
labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
pie.set_facecolor('#013777')
wedges, texts, autotexts = plt.pie(x=sen, autopct="%.1f%%", colors=colors, explode=[0.05]*3, labels=labels, pctdistance=0.5, textprops=font)
for text in autotexts:
    text.set_color('#012767')
font['size'] = 16
plt.title("TWEET SENTIMENT", fontdict=font)
pie.savefig('pie.png')

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 16,
        }
bars = data.text.groupby(data.datetime.dt.hour).count()
# bars.shape = (int(len(bars)/2), 2)
print(bars)
tph = bars.values
title = 'Number of Tweets per Hour'
fig, ax = plt.subplots(figsize=(12,8))
fig.patch.set_facecolor('#013777')
ax.set_facecolor('#013777')
ax.bar(bars.index, tph, color='#c81f2d')
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.4)
# Remove x,y Ticks
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.set_xticks(bars.index, minor=False)
ax.set_yticks(range(0,4001,1000), minor=False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
a=gca()
a.set_xticklabels(a.get_xticks(), font)
a.set_yticklabels(a.get_yticks(), font)
font['size'] = 20
plt.xlabel('HOUR', fontdict=font)
plt.ylabel('TWEETS', fontdict=font)
font['size'] = 25
plt.title('NUMBER OF TWEETS DURING EACH HOUR', fontdict=font)
# plt.show()
plt.savefig('hours.png')