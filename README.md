# TF-IDF
Enable users to tune the parameters like: 
  1.max_df: If this word appear more than 50% of all documents, this word will be ignored.
  2.topn: Only keep top N words based on TF-IDF score for each document(review).
  #topn words here are like hashtag for each document
  
Count the frequency of these topnwords among all documents-->Dictionary:
  doctor:110
  patient:105
  observation:100
  careful:78

Generate wordCloud based on dictionary get above.
