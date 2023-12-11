""" nltk.download('puntk')
nltk.download('stopwords') """
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from natsort import natsorted
import os 
import nltk
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

stop_words = stopwords.words('english')
stop_words.remove('to')
stop_words.remove('in')
stop_words.remove('where')

#READ documents And TOKENIZATION And STEEMING
ps = PorterStemmer()
document_of_terms = []
document_with_stemming = []
file_name = natsorted(os.listdir('DocumentCollection')) #to sort files in folder
for file in file_name :
    #1.1# Read 10 Documents
    with open(f'DocumentCollection/{file}', 'r') as f:
        document = f.read()
    #1.2# apply tokenizing
    tokenizing = word_tokenize(document)
    for word in tokenizing : 
        if word not in stop_words :
            #1.3# apply stemming
            stemming = ps.stem(word)
        document_with_stemming.append(stemming)
    document_of_terms.append(document_with_stemming)
    document_with_stemming = []
#print(document_of_terms)

# positional index
document_id = 1
positional_index = {}
for document in document_of_terms:
    for position, term in enumerate(document):
        if term in positional_index:
            positional_index[term][0] = positional_index[term][0] + 1
            if document_id in positional_index[term][1]:
                positional_index[term][1][document_id].append(position)
            else:
                positional_index[term][1][document_id] = [position]
        else:
            positional_index[term] = []
            positional_index[term].append(1)
            positional_index[term].append({})
            positional_index[term][1][document_id] = [position]
    document_id = document_id + 1
sorted_posIndex = dict(sorted(positional_index.items()))
print("POSITIONAL INDEX IS :")
print(sorted_posIndex)
print()
print()

#2.5# phrase query
def phrase_query(q):
    outer_array = [[] for i in range(10)]
    for wordd in q.split():
        word_after_stem = ps.stem(wordd)
        if word_after_stem in positional_index.keys():  
            for key in positional_index[word_after_stem][1].keys():
                if outer_array[key-1] != []:
                    if outer_array[key-1][-1] == positional_index[word_after_stem][1][key][0] - 1:
                        outer_array[key-1].append(positional_index[word_after_stem][1][key][0])
                else:
                    outer_array[key-1].append(positional_index[word_after_stem][1][key][0])
    positions = []
    for position, arr in enumerate(outer_array, start=1):
        if len(arr) == len(q.split()):
            positions.append('doc'+str(position))
    return positions

#boolean query
def process_boolean_query(query, document_indices):
    terms = query.split()
    result = []

    if 'AND' in terms:
        # AND operation
        index_sets = [set(document_indices[term]) for term in terms if term != 'AND']
        result = list(set.intersection(*index_sets))
    elif 'OR' in terms:
        # OR operation
        index_sets = [set(document_indices[term]) for term in terms if term != 'OR']
        result = list(set.union(*index_sets))
    else:
        # Single term
        result = document_indices.get(terms[0], [])

    return [f"document {idx}" for idx in result]

#tf
all_terms = []
for doc in document_of_terms:
    for word in doc:
        all_terms.append(word)
def term_frequency(doc):
    #put each term in dictionary form with value 0
    word_count = dict.fromkeys(all_terms, 0) 
    for word in doc:
        #increment the value of each term in dictionary
        word_count[word] += 1
        # return dictionary of each document
    return word_count
# convert dictionay form into columns form
tf = pd.DataFrame(term_frequency(document_of_terms[0]).values(), index = term_frequency(document_of_terms[0]). keys())
for i in range(1, len(document_of_terms)):
    tf[i] = term_frequency(document_of_terms[i]).values()
#change name of column
tf.columns = ['doc' + str(i) for i in range(1, 11)] 
sorted_tf = tf.sort_index()
print("TERM FREQUENCY IS:")
print(sorted_tf)
print()
print()

#w_tf weighted term frequency 1 + log10(tf)
def w_tf(x):
    if x > 0:
        return math.log10(x) + 1
    else:
        return 0
wt_tf = tf.copy()
for i in range(1, len(document_of_terms) + 1):
    #replace each raw with w_tf
    wt_tf['doc'+ str(i)] =  wt_tf['doc'+str(i)].apply(w_tf) 
sorted_wt_tf = wt_tf.sort_index()
print("WT_TF IS:")
print(sorted_wt_tf)
print()
print()

#2.3# IDF
df_and_IDF = pd.DataFrame(columns=('d_f','idf'))
for i in range(len(tf)):
    doc_freq = tf.iloc[i].values.sum()
    df_and_IDF.loc[i, 'd_f'] = doc_freq
    df_and_IDF.loc[i, 'idf'] = math.log10(10/float(doc_freq))
df_and_IDF.index = tf.index
sorted_df_and_IDF = df_and_IDF.sort_index()
print("DF AND IDF IS :")
print(sorted_df_and_IDF)
print()
print()

# TF-IDF
tf_idf = tf.multiply(df_and_IDF['idf'], axis = 0)
sorted_tf_idf = tf_idf.sort_index()
print("TF_IDF IS :")
print(sorted_tf_idf)
print()
print()

#doc_length
import numpy as np
doc_length = pd.DataFrame()
def document_length(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())
for column in tf_idf.columns:
    doc_length.loc[0, column + '_len'] = document_length(column)
print("DOCUMENT LENGTH IS :")
print(doc_length)
print()
print()

#NORMALIZATION TF-IDF divided by DOC_LENGTH
normalize = pd.DataFrame()
def get_normalize(col, x):
    try:
        return x / doc_length[column + '_len'].values[0]
    except:
        return 0
for column in tf_idf.columns:
    normalize[column] = tf_idf[column].apply(lambda x: get_normalize(column, x))
print("NORMALIZTION")
sorted_normalize = normalize.sort_index()
print(sorted_normalize)
print()
print()
##########################################QUERY####################################
q = 'fools fear'
def wtf(x):
    if x > 0:
        return 1 + math.log10(x)
    else:
        return 0
def insert_query(q):
    document_found = phrase_query(q)
    if document_found == []:
        print("Not found")
        return 0
    query = pd.DataFrame(index = normalize.index)
    x = []
    for word in q.split():
        x.append(ps.stem(word))
    #tf of query
    query['tf'] = [x.count(term) if term in x else 0 for term in list(normalize.index)]
    query['w_tf'] = query['tf'].apply(lambda x : int(wtf(x)))
    query['idf'] = df_and_IDF['idf'] * query['w_tf']
    query['tf-idf'] = query['tf'] * query['idf']
    query['normalized'] = 0
    product = normalize.multiply(query['w_tf'], axis = 0)
    for i in range(len(query)):
        #normalizarion of each term in query
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
    #normailze of each term in document * normalize of each term in query
    product2 = product.multiply(query['normalized'], axis = 0)
    print("Query Details")  
    print(query)
    scores = {}
    print(document_found)
    print(product2)
    #get product2 (normalize of document1*normalize of query, ormalize of document2*normalize of query)
    for col in document_found:
        #sum of column1 that was normalize of doc1 * normalize of query
        scores[col] = product2[col].sum()
    #result of product normalize of document * normalize of query
    result = product2[list(scores.keys())].loc[x]
    print()
    print('Product (query*matched doc)')
    print(result)
    print()
    print("PRODUCT SUM")
    print(result.sum())
    print()
    final_scored = sorted(scores.items(), key = lambda x :x[1], reverse=True)
    print("COSINE SIMILARITY of each document IS:")
    print(final_scored)
    print()
    print("Matched documents ordered with high SIMILARITY:")
    for doc in final_scored:
        print(doc[0], end=' ')
    print()

insert_query(q)

