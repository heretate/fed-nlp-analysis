from tkinter import X
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import re
import datetime as dt
from matplotlib.pyplot import get
from numpy import average
import pandas as pd
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
import heapq

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

chairpersons = pd.DataFrame(
    data=[
          ["Bernanke", "Ben", dt.datetime(2006,2,1), dt.datetime(2014,1,31)], 
          ["Yellen", "Janet", dt.datetime(2014,2,3), dt.datetime(2018,2,3)],
          ["Powell", "Jerome", dt.datetime(2018,2,5), dt.datetime(2026,5,15)]],
    columns=["Surname", "FirstName", "FromDate", "ToDate"])

stop_words = set(stopwords.words('english'))
ignore_keywords = set(["powell", "chair", "yellen"])

def get_fomc_dates(from_year, to_year):
    base_url='https://www.federalreserve.gov'
    calendar_url='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
    req = Request(calendar_url, headers={'User-Agent': 'Mozilla/5.0'})
    fomc_meetings_socket = urlopen(req)
    soup = BeautifulSoup(fomc_meetings_socket, 'html.parser')
    statements = soup.find_all('a', href=re.compile('^/newsevents/pressreleases/monetary\d{8}a.htm'))
    links = [statement.attrs['href'] for statement in statements] 

    fomc_potential_dates = []
    for link in links:
        m = re.search('/newsevents/pressreleases/monetary(.+?)a.htm', link)
        if m:
            date = dt.datetime.strptime(m.group(1), "%Y%m%d") 
            if date >= dt.datetime(from_year, 1,1) and date <= dt.datetime(to_year, 12, 1):
                fomc_potential_dates.append(m.group(1))

    
    fomc_potential_dates = sorted(fomc_potential_dates)
    
    return fomc_potential_dates

def get_fomc_text(str_date, fomc_doc, type='pdf'):
    
    if fomc_doc == "statement":
        if type == 'pdf':
            url = 'https://www.federalreserve.gov/monetarypolicy/files/monetary{}a1.pdf'.format(str_date)
        elif type == 'html': 
            url = 'https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm'.format(str_date)    
        else:
            raise ValueError(f"Unrecognized format type {type}")
    elif fomc_doc == "minutes":
        
        if type == 'pdf':
            url = 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes{}.pdf'.format(str_date)
        elif type == 'html': 
            url = 'https://www.federalreserve.gov/monetarypolicy/fomcminutes{}.htm'.format(str_date)    
        else:
            raise ValueError(f"Unrecognized format type {type}")
        
    elif fomc_doc == "press":
        url = 'https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{}.pdf'.format(str_date)
    else:
        raise ValueError("Unrecognized document type")
    try:
    
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        if type == 'pdf':
            file_output = []
            file = urlopen(req)
            pdf_reader = PyPDF2.PdfFileReader(BytesIO(file.read()))
            for i in range(pdf_reader.numPages):
                page = pdf_reader.getPage(i)
                file_output.append(page.extractText())
        else:
            soc = urlopen(req)
            article = BeautifulSoup(soc, 'html.parser')
            for fn in article.find_all('a', {'name': re.compile('fn\d')}):
                fn.decompose()
            paragraphs = article.findAll('p')
            file_output = [paragraph.get_text().strip() for paragraph in paragraphs]
    except HTTPError:
        file_output = []
        print(f"Couldn't retrieve data for date {str_date} and file type {fomc_doc}")
    return file_output


def get_chairperson(x):
    '''
    Return a tuple of chairperson's Fullname for the given date x.
    '''
    
    chairperson = chairpersons.loc[chairpersons['FromDate'] <= x].loc[x <= chairpersons['ToDate']]
    return list(chairperson.FirstName)[0] + " " + list(chairperson.Surname)[0]
    

def parse_press_conference(document, date, debug=False):
    # Expect a list of list of lists
    if not document:
        if debug:
            print(f"Press Conference for {str(date)} not found")
        return []
    
    merged_pages = []
    for page in document:
        concat_segments = "".join(page)
        merged_pages.append(concat_segments)

    final_string = "\n".join(merged_pages)

    paragraphs = final_string.strip()
    paragraphs = paragraphs.split("\n")

    section = -1
    article_sections = []
    for article in paragraphs:
        if len(re.findall(r'[A-Z]', article[:10])) > 5:
            section += 1
            article_sections.append("")
        if section >= 0:
            if not re.search('^(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', article.lower()):
                if not re.search('^page', article.lower()):
                    article_sections[section] += '\n\n' + article

    if debug:
        for section in article_sections: 
            input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', section)
            section.strip()
            print(len(input_words))
            print(section)
            print("--------------------------------------------------")
    
    
    # Filter out any speaker that isn't the chair
    
    final_sections = []
    
    chair = get_chairperson(date)
    chair = chair.split(" ")[1].upper()
    for section in article_sections:
        if re.search(chair, section[:20]):
            final_sections.append(section)
    
    return final_sections


def parse_statements(document, date, debug=False):
    # Expect a list of list of lists
    if not document:
        if debug:
            print(f"Statement for {str(date)} not found")
        return []

    paragraphs = document.copy()


    articles = []
    for article in paragraphs:
        article = article.lower()
        if len(article.strip()) < 100:
            continue
        if not re.search('(for release at|-#-)', article):
            article = article.lower()
            articles.append(article)
            
    if debug:
        for section in articles: 
            input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', section)
            section.strip()
            print(len(input_words))
            print(section)
            print("--------------------------------------------------")

    return articles
  
def parse_minutes(document, date, debug=False):
    # Expect a list of list of lists
    
    if not document:
        if debug:
            print(f"Minutes for {str(date)} not found")
        return []
    
    sentiment_keywords = get_sentiment_dictionary()
    sentiment_keywords = sentiment_keywords["Positive"].union(sentiment_keywords["Negative"])
    paragraphs = document.copy()
    section = 0
    articles = []
    for article in paragraphs:
        article = article.lower()
        article = article.strip("o•_")
        if len(article) < 100:
            continue
        
        input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', article)
        if not set(input_words).intersection(sentiment_keywords):
            continue
        articles.append(article)

    if debug:
        for section in articles: 
            input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', section)
            section.strip()
            print(len(input_words))
            print(section)
            print("--------------------------------------------------")

    return articles

def tokenize_paragraph(paragraph, remove_stop_words=True, ignore_dict={}):
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(paragraph)
    filtered_paragraph = [w.lower() for w in word_tokens if not w.lower() in stop_words.union(ignore_keywords.union(ignore_dict)) and not w.isnumeric()]
    return filtered_paragraph

def get_sentiment_dictionary():
    path = 'data/Loughran-McDonald_MasterDictionary_1993-2021.csv'
    
    data = pd.read_csv(path)
    
    positive = data[data['Positive'] != 0]
    negative = data[data['Negative'] != 0]
    s_dict = {'Positive':set(), 'Negative':set()}
    s_dict['Positive'] = set(positive["Word"].str.lower())
    
    s_dict['Negative'] = set(negative["Word"].str.lower())
        
    return s_dict

negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
          "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
          "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
          "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
          "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]

def negated(word):
    """
    Determine if preceding word is a negation word
    """
    if word.lower() in negate:
        return True
    else:
        return False
    
def get_article_sentiment(article, dict):
    pos_count = 0
    neg_count = 0

    pos_words = []
    neg_words = []
    
    ignored_words = set(['question'])
 
    input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', article.lower())
    word_count = len(input_words)
     
    for i in range(0, word_count):
        if input_words[i] in ignored_words:
            continue
        if input_words[i] in dict['Negative']:
            neg_count += 1
            neg_words.append(input_words[i])
        if input_words[i] in dict['Positive']:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 0:
                pos_count += 1
                pos_words.append(input_words[i])
    
    results = [pos_count, neg_count, pos_words, neg_words]
 
    return results

def convert_to_dataframe(documents, time_frame, doc_type, other_data=None):
    data = []
    for i, date in enumerate(time_frame):
        if documents[i]:
            for section in documents[i]:
                row = [date, doc_type, section]
                data.append(row)
    
    data = pd.DataFrame(data, columns = ["date", "type", "content"])
    
    return data
            

def get_lexical_sentiment(data, dict):
    sentiment = []
    for article in data["content"]:
        sentiment_row = get_article_sentiment(article, dict)
        sentiment.append(sentiment_row)
    
    sentiment_data = pd.DataFrame(sentiment, columns = ['n_pos_words', 'n_neg_words', 'pos_words', 'neg_words'])
    
    new_data = pd.concat([data, sentiment_data], axis=1)
    
    return new_data

def get_top_sentiment_text(data, n=10, save_local=False):
    
    def agg_rows(col):
        agg_dict = defaultdict(int)
        for row in col:
            for word in row:
                agg_dict[word] += 1
                
        return agg_dict
    agg_data = data[['date', 'pos_words', 'neg_words']].groupby('date').agg(lambda x: agg_rows(x)).reset_index()
    
    pos_top = []
    neg_top = []
    for index, row in agg_data.iterrows():
        pos = heapq.nlargest(n, row['pos_words'], key=row['pos_words'].get)
        neg = heapq.nlargest(n, row['neg_words'], key=row['neg_words'].get)
        pos_top.append(pos)
        neg_top.append(neg)
    
    pos_top = pd.DataFrame(pos_top).transpose()
    neg_top = pd.DataFrame(neg_top).transpose()
    pos_top.columns = agg_data['date']
    neg_top.columns = agg_data['date']
    
    if save_local:
        pos_top.to_excel("Top Positive Words.xlsx")
        neg_top.to_excel("Top Negative Words.xlsx")
    return pos_top, neg_top

def get_top_text(data, n=10, save_local=False):
    
    additional_ignored_texts = set(['think', 'see', 'would', 'know', 'say', 'get', 'going', 'well', 'percent', 'really', 'much', 'look', 'back', 'one', 'year', 'u', 'also', 'committee', 'participants'])
    
    data = data[['date', 'content']].copy()
    data['content'] = data['content'].map(lambda x: tokenize_paragraph(x.lower(), remove_stop_words=True, ignore_dict = additional_ignored_texts))
    
    
    def agg_rows(col):
        agg_dict = defaultdict(int)
        for row in col:
            for word in row:
                agg_dict[word] += 1
                
        return agg_dict
    agg_data = data.groupby('date').agg(lambda x: agg_rows(x)).reset_index()
    
    top_text = []
    for index, row in agg_data.iterrows():
        top = heapq.nlargest(n, row['content'], key=row['content'].get)
        top_text.append(top)
    
    top_text = pd.DataFrame(top_text).transpose()

    top_text.columns = agg_data['date']
    
    if save_local:
        top_text.to_excel("Top Used Words.xlsx")
    return top_text


def get_target_average(from_year, to_year):
    lower_target = pd.read_csv("data/DFEDTARL.csv")
    upper_target = pd.read_csv("data/DFEDTARU.csv")
    average_target = lower_target[['DATE']]
    average_target['target'] = (lower_target['DFEDTARL'] + upper_target['DFEDTARU']) / 2
    average_target.columns = ['date', 'target']
    average_target['date'] = pd.to_datetime(average_target['date'])
    average_target = average_target[(average_target['date'] >= dt.datetime(from_year,1,1)) & (average_target['date'] <= dt.datetime(to_year,12,31))]
        
    return average_target
    
def get_target_direction(dates):
    lower_target = pd.read_csv("data/DFEDTARL.csv")
    upper_target = pd.read_csv("data/DFEDTARU.csv")
    average_target = lower_target[['DATE']]
    average_target['target'] = (lower_target['DFEDTARL'] + upper_target['DFEDTARU']) / 2
    average_target.columns = ['date', 'target']
    average_target['date'] = pd.to_datetime(average_target['date'])
    average_target.set_index('date', inplace=True)
    
    directions = []
    for date in dates:
        change_date = date + dt.timedelta(days=1)
        direction = average_target.loc[[change_date], 'target'][0] - average_target.loc[[date], 'target'][0] # Datetime Index does not accept non-range accessing
        
        if direction > 0:
            directions.append({'date':date, 'target':1})
        elif direction == 0:
            directions.append({'date':date, 'target':0})
        else:
            directions.append({'date':date, 'target':-1})

    data = pd.DataFrame.from_records(directions)
        
    return data
    

def get_clean_train_data(data, metadata):
    agg_data = data.copy()
    agg_data['content'] = agg_data['content'].map(lambda x: tokenize_paragraph(x))
    agg_data['content'] = agg_data['content'].map(lambda x: [stemmer.stem(w) for w in x])
    agg_data['content'] = agg_data['content'].map(lambda x: [lemmatizer.lemmatize(w) for w in x])
    agg_data['content'] = agg_data['content'].map(lambda x: x if len(x) > 20 else pd.NA)
    agg_data = agg_data.dropna()
    agg_data = agg_data[['date', 'type', 'content']]
    agg_data = agg_data.merge(metadata, how='left', left_on='date', right_on='date')
    
    train_x = []
    train_y = []
    for index, row in agg_data.iterrows():
        paragraphs = get_split_paragraphs(row['content'], threshold=200)
        train_x = train_x + paragraphs
        train_y = train_y + [row['target']] * len(paragraphs)
    return train_x, train_y

def get_split_paragraphs(paragraph, threshold=200):
    paragraphs = []
    i = 0
    while i < len(paragraph):
        upper = min(len(paragraph), i + threshold)
        lower = i
        if upper - i < threshold * 0.1:
            lower = max(0, int((i - threshold * 0.25)))
            next_para = paragraph[lower:len(paragraph)]
            i = len(paragraph)
        else:
            next_para = paragraph[lower:upper]
            i += int(threshold / 2)
        paragraphs.append(next_para)
    
    return paragraphs

def get_vocab(data):
    vocab = set()
    
    for index, row in data.iterrows():
        for word in row['content']:
            vocab.add(word)
            
    vocab_dict = {}
    
    for i, word in enumerate(vocab, 1):
        vocab_dict[word] = i
    return vocab_dict






