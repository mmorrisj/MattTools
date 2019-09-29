# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:06:01 2019

@author: mmorr
"""
# import json
import pandas as pd
import matplotlib.pyplot as plt
# import spacy
# from spacy import displacy
import en_core_web_sm
# from pathlib import Path
import heapq
import re
import unicodedata
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize.toktok import ToktokTokenizer
# from contractions import CONTRACTION_MAP
# from nltk.stem import PorterStemmer
# from nltk import sent_tokenize, word_tokenize
import operator
import seaborn as sns
import nltk
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import multiprocessing as mp

from multiprocessing import Lock, Process, Queue, current_process

import time


# start = time.time()
## run your code
# end = time.time()
#
# elapsed = end - start

def clean_content(content):
    text = content.replace('\n', '').replace('_', '').replace('<pre>', '').replace('</pre>', '').replace('\n',
                                                                                                         '').replace(
        '|', '').strip()

    text = remove_boilerplate(text)

    text = remove_accented_chars(text)

    #    text=expand_contractions(text)

    #    text = re.sub('[^a-zA-Z]', ' ', text )

    text = re.sub(r'\s+', ' ', text)

    #    text=remove_special_characters(text)

    #    text=(lemmatize_text(text))

    return text


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def build_summaries(df, id_field, content_field):
    summary_df = []
    count = 0
    total = 0
    print('processing summaries...')

    for doc_id, doc in zip(df[id_field], df[content_field]):

        article_text = clean_content(doc)

        sentence_list = nltk.sent_tokenize(article_text)

        stopwords = nltk.corpus.stopwords.words('english')

        word_frequencies = {}

        for word in nltk.word_tokenize(article_text):

            if word not in stopwords:

                if word not in word_frequencies.keys():

                    word_frequencies[word] = 1

                else:

                    word_frequencies[word] += 1

        try:
            maximum_frequncy = max(word_frequencies.values())
        except:
            summary_df.append({'title': doc_id, 'summary': 'no_summary'})
            continue

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

        sentence_scores = {}

        for sent in sentence_list:

            for word in nltk.word_tokenize(sent.lower()):

                if word in word_frequencies.keys():

                    if len(sent.split(' ')) < 30:

                        if sent not in sentence_scores.keys():

                            sentence_scores[sent] = word_frequencies[word]

                        else:

                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)

        summary_df.append({'title': doc_id, 'summary': str(summary)})
        count += 1
        total += 1

        if count == 100:
            print(str(len(df.title) - total) + ' documents remain.')
            count = 0
    return pd.DataFrame(summary_df)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def build_labels(df, id_field, content_field):
    print('processing labels...')

    nlp = en_core_web_sm.load()

    labels_list = []

    count = 0
    total = 0

    for title, doc in zip(df[id_field], df[content_field]):

        article_text = clean_content(doc)

        doc = nlp(article_text)

        labels_list.append({'title': title,

                            'Persons': [ent for ent in doc.ents if ent.label_ == 'PERSON'],

                            'Institutions': [ent for ent in doc.ents if ent.label_ == 'GPE'],

                            'Organizations': [ent for ent in doc.ents if ent.label_ == 'ORG'],

                            'Verbs': [ent for ent in doc if ent.pos_ == 'VERB']
                            })
        total += 1
        count += 1
        if count == 100:
            print(str(len(df[id_field]) - total) + ' documents remain.')

            count = 0

    return pd.DataFrame(labels_list)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def label_text(text):
    nlp = en_core_web_sm.load()

    labels_df = []

    article_text = clean_content(text)

    doc = nlp(article_text)

    labels_df.append({'Persons': [ent for ent in doc.ents if ent.label_ == 'PERSON'],

                      'Institutions': [ent for ent in doc.ents if ent.label_ == 'GPE'],

                      'Organizations': [ent for ent in doc.ents if ent.label_ == 'ORG']})
    return pd.DataFrame(labels_df)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
#    
#    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
#                                      flags=re.IGNORECASE|re.DOTALL)
#    def expand_match(contraction):
#        match = contraction.group(0)
#        first_char = match[0]
#        expanded_contraction = contraction_mapping.get(match)\
#                                if contraction_mapping.get(match)\
#                                else contraction_mapping.get(match.lower())                       
#        expanded_contraction = first_char+expanded_contraction[1:]
#        return expanded_contraction
#        
#    expanded_text = contractions_pattern.sub(expand_match, text)
#    expanded_text = re.sub("'", "", expanded_text)
#    return expanded_text

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'

    text = re.sub(pattern, '', text)

    return text


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()

    text = ' '.join([ps.stem(word) for word in text.split()])

    return text


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def lemmatize_text(text):
    nlp = en_core_web_sm.load()

    text = nlp(text)

    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    return text


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def summarize_text(text, num_sentences):
    #    article_text=clean_content(text)

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', text)

    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(formatted_article_text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}

    for word in nltk.word_tokenize(formatted_article_text):

        if word not in stopwords:

            if word not in word_frequencies.keys():

                word_frequencies[word] = 1

            else:

                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    #    summary_df.append({'title':title,'summary':'no_summary'})

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_scores = {}

    for sent in sentence_list:

        for word in nltk.word_tokenize(sent.lower()):

            if word in word_frequencies.keys():

                if len(sent.split(' ')) < 30:

                    if sent not in sentence_scores.keys():

                        sentence_scores[sent] = word_frequencies[word]

                    else:

                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

    #    summary_df.append({'title':title,'summary':str(summary)})

    return summary


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def visualize_labels(label_df):
    print('visualizing labels...')

    persons = [indiv.text for item in label_df.Persons for indiv in item]

    instit = [instit.text for item in label_df.Institutions for instit in item]

    orgs = [org.text for item in label_df.Organizations for org in item]

    sorted_persons = count_frequency(persons)

    name_list = [k for k, v in sorted_persons]

    num_of_names = [v for k, v in sorted_persons]

    sorted_instits = count_frequency(instit)

    instit_list = [k for k, v in sorted_instits]

    num_of_instits = [v for k, v in sorted_instits]

    sorted_orgs = count_frequency(orgs)

    org_list = [k for k, v in sorted_orgs]

    num_of_orgs = [v for k, v in sorted_orgs]

    sns.set(style='dark')

    plt.figure(figsize=[8, 15])

    plt.subplot(311)

    plt.barh(name_list[-10:], num_of_names[-10:], color='darkgrey')

    plt.title('Top 10 Persons Mentioned in Documents')

    plt.subplot(312)

    plt.barh(instit_list[-10:], num_of_instits[-10:], color='darkgrey')

    plt.title('Top 10 Governments Mentioned in Documents')

    plt.subplot(313)

    plt.barh(org_list[-10:], num_of_orgs[-10:], color='darkgrey')
    #    _label_barh(org_list[-10:],[x/sum(num_of_orgs[-10:]) for x in num_of_orgs[-10:]],color='darkgrey',value_list)
    plt.title('Top 10 Organizations Mentioned in Documents')

    plt.savefig('./Label Charts.png')


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def count_frequency(list):
    word_frequencies = {}

    for word in list:
        if word not in word_frequencies.keys():

            word_frequencies[word] = 1

        else:

            word_frequencies[word] += 1

    sorted_list = sorted(word_frequencies.items(), key=operator.itemgetter(1))

    return sorted_list


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(
        y == ys[0] for y in ys)  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', color=color, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def _label_barh(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long
    """
    max_x_value = ax.get_xlim()[1]

    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance

        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def remove_boilerplate(text):
    bp = ['The Project Gutenberg EBook of',
          'This eBook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever',
          'You may copy it, give it away or re-use it under the terms',
          'with this eBook or online',
          'START OF THIS PROJECT',
          'Copyright',
          'This Web site includes information about Project Gutenberg',
          'how to help produce our new eBooks',
          'Most people start at our Web site',
          'eBooks are often created from several printed',
          'keep eBooks in compliance with',
          'library of electronic works',
          'originator of the Project Gutenberg',
          'Project Gutenberg',
          'Gutenberg',
          'including checks, online payments and credit card',
          'statements concerning tax treatment of donations received',
          'accepting unsolicited donations from donors',
          'Foundation is committed to complying',
          'do not solicit donations in locations',
          'are particularly important to maintaining tax exempt',
          'volunteers and employees are scattered',
          'Illustration:',
          'You may copy it',
          ]

    good_sentences = [sentence for boiler in bp for sentence in nltk.sent_tokenize(text) \
                      if boiler.lower().replace(' ', '').strip() not in sentence.lower().replace(' ', '').strip() \
                      if boiler not in sentence]

    #    bad_sentences=[sentence for boiler in bp for sentence in nltk.sent_tokenize(text)\
    #                   if boiler.lower().replace(' ','').strip() in sentence.lower().replace(' ','').strip()\
    #                   if boiler in sentence if sentence not in bad_sentences]

    content = ' '.join(good_sentences).strip()

    return content


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
def folder(loc):
    base = 'C:\\Users\\mmorr\\Desktop\\code\\'

    if loc == 'base':
        folder = base

    if loc == 'c':
        folder = base + 'Congress\\'

    return folder


def filename(loc, name, file_type='xlsx'):
    if file_type == 'txt':
        return folder(loc) + str(name) + '.txt'
    else:
        return folder(loc) + str(name) + '.xlsx'


def ingest(name, loc='base', sheet_name='Sheet1', skip_rows=0, file_type='xlsx'):
    if file_type == 'txt':

        file = open(filename(loc, name, file_type='txt', mode='r'))

        text = file.read()

        return text
    else:

        df = pd.read_excel(filename(loc, name), sheet_name=sheet_name, skip_rows=skip_rows)

        return df


def export(df, name, loc='c', sheet_name='Sheet1', skip_rows=0):
    print("Exporting to:" + str(filename))

    return df.to_excel(filename(loc, name), sheet_name=sheet_name, skip_rows=skip_rows)


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
def build_word_frequencies(text):
    word_frequencies = {}

    for word in nltk.word_tokenize(text):

        if word not in stopwords:

            if word not in word_frequencies.keys():

                word_frequencies[word] = 1

            else:

                word_frequencies[word] += 1

    return word_frequencies


# fig, ax = plt.subplots(figsize=[8,15])
# plt.subplot(311)
# name_horizontal_bars = ax.barh(name_list[-10:], [x/sum(num_of_names[-10:]) for x in num_of_names[-10:]], align="center")
# value_format = "{:.1%}"  # displaying values as percentage with one fractional digit
# label_bars(ax, name_horizontal_bars, value_format)
# plt.subplot(312)
##fig2, ax2 = plt.subplots()
# org_horizontal_bars = ax.barh(org_list[-10:], [x/sum(num_of_orgs[-10:]) for x in num_of_orgs[-10:]], align="center")
# value_format = "{:.1%}"  # displaying values as percentage with one fractional digit
# label_bars(ax2, org_horizontal_bars, value_format)
# plt.subplot(313)
##fig3, ax3 = plt.subplots()
# instit_horizontal_bars = ax.barh(instit_list[-10:], [x/sum(num_of_instits[-10:]) for x in num_of_instits[-10:]], align="center")
# value_format = "{:.1%}"  # displaying values as percentage with one fractional digit
# label_bars(ax3, instit_horizontal_bars, value_format)
#
# plt.savefig('./Label Charts.png')

def score_sentences(sentence_list, word_frequencies):
    sentence_scores = {}

    for sentence in sentence_list:

        for sent in sentence:

            for word in nltk.word_tokenize(sent.lower()):

                for item in word_frequencies:

                    if word in item.keys():

                        if len(sent.split(' ')) < 30:

                            if sent not in sentence_scores.keys():

                                sentence_scores[sent] = item[word]

                            else:

                                sentence_scores[sent] += item[word]
    return sentence_scores


def load_data(name, loc, content_field='content'):
    df = ingest(name, loc)

    df['clean_content'] = [clean_content(x) for x in df[content_field]]



# import time
#
# import queue # imported for using queue.Empty exception


def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            print(task)
            tasks_that_are_done.put(task + ' is done by ' + current_process().name)
            time.sleep(.5)
    return True


def test_no_mp(df, elapsed_time):
    start = time.time()

    df['test'] = [clean_content(x) for x in df.content]
    end = time.time()

    elapsed = end - start

    print(str(elapsed) + ' seconds to run ' + str(len(df)) + ' documents...')
    elapsed_time.append(elapsed)
    return round(Average(elapsed_time))


# pool = mp.Pool(mp.cpu_count())
# results = pool.apply(test_mp,args=(short_senate,elapsed_time_mp))
def test_mp(df, elapsed_time_mp, pool):
    start = time.time()

    results = pool.map(clean_content, [x for x in df.content])

    pool.close()

    end = time.time()

    elapsed = end - start

    elapsed_time_mp.append(elapsed)
    print(str(round(Average(elapsed_time_mp))) + ' seconds to run ' + str(len(df)) + ' documents...')
    return results


def Average(elapsed_time):
    return sum(elapsed_time) / len(elapsed_time)
