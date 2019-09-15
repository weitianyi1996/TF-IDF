#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#treat each doctor(hp_id)'s review as a document
#doc_review['review_corpus']=doc_review['review_corpus'].apply(fun)
import re

def string_preprocessing(s):
    stripped=re.sub('|','',s)
    stripped=re.sub("<!--?.*?-->","",s)
    stripped=re.sub('[\s+]',' ',s)

    stripped=stripped.strip()#remove start and end white empty space

    return stripped

#creating vocabulary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#get the whole text
docs=doc_review.loc[doc_review['spec_comb']=='Internal Medicine','review_corpus'].tolist()

#ignore words that appear in 50% of documents, eliminate stop words
def tune_maxdf(n,docstolist):
    cv=CountVectorizer(max_df=n,lowercase=True,stop_words=stopwords)
    word_count_vector=cv.fit_transform(docstolist) #create a vocabulary of words for docs
    print("""the matrix shape is:"""+ str(word_count_vector.shape))

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector) #get the IDF score ready for the vocabulary

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

def topNwords(doc,topn=3):
    #doc is each doctor's review
    """
    This function will return a topn words dictionary with their TF-IDF score
    for each record of a specific column('review_corpus')
    """
    # you only needs to do this once, this is a mapping of index to
    feature_names=cv.get_feature_names()

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,topn)

    dic={k:keywords[k] for k in keywords}

    return dic

#each female doctor top n words based on TF-IDF
def reviewToWordcloud(review_col,topn=3):
    """
    This function will return topn words based on TFIDF score in each doctor's review,
    count frequency for these words for all doctors' reviews,
    generate wordCloud based on this dictionary.
    """
    f_dic={}
    for dict in review_col.apply(lambda x:topNwords(x,topn)).tolist():
        for x in dict.keys():
            if x not in f_dic:
                f_dic[x]=1
            else:
                f_dic[x]+=1

    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate_from_frequencies(f_dic)
    # plot the WordCloud image
    plt.figure(figsize = (12, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()


# In[ ]:


doc_review['review_corpus']=doc_review['review_corpus'].apply(string_preprocessing)
tune_maxdf(0.25,docs)
reviewToWordcloud(m_int_med,5)
