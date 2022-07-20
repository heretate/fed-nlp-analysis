
from turtle import pos
import pandas as pd
import matplotlib.pyplot as plt
import re


def visualize_basic_sentiment(data, rates,title="Sentiment", proportionate=False):
    
    sentiment_data = data[data["n_pos_words"] + data["n_neg_words"] != 0]
    
    sentiment_data = sentiment_data[['date', "n_pos_words", "n_neg_words"]]
    sentiment_data = sentiment_data.groupby('date').sum()
    
    if proportionate:
        sentiment_data["total"] = sentiment_data['n_pos_words'] + sentiment_data['n_neg_words']
        sentiment_data['n_pos_words'] = sentiment_data['n_pos_words'] / sentiment_data["total"] * 100
        sentiment_data['n_neg_words'] = sentiment_data['n_neg_words'] / sentiment_data["total"] * 100
    
    sentiment_data = sentiment_data.merge(rates, how='left', left_on='date', right_on='date')
    
    df = pd.DataFrame({"Positive":sentiment_data['n_pos_words'],"Negative":sentiment_data['n_neg_words']})
    fig, ax1 = plt.subplots(figsize=(12,8))
    ax2 = ax1.twinx()
    ax1.plot('date', 'n_pos_words', data=sentiment_data)
    ax1.plot('date', 'n_neg_words', data=sentiment_data)
    #ax1 = df.plot.line(color=["SkyBlue","IndianRed"], rot=0, title=title)
    ax2.plot('date', 'target', data=sentiment_data, color='green')
    ax2.set_ylim([-1, 3])
    ax2.set_ylabel("Target Average (%)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Tone Proportion (%)")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    ax1.set_title(title)
    fig.autofmt_xdate(rotation=-45)
    
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(times.strftime("%b %d %Y")))
    plt.show()
    
    
    
def visualize_text(text_list):
    for text in text_list:
        print(text)
        print(f"--------------------------------------------------{len(text)}")