"""

Naive Bayes Classification
Supervised Learning - Classification technique
What is does and how - https://www.youtube.com/watch?v=EGKeC2S44Rs
Maths behind - Very Simple Probability - Bayes Theorem with 'naive' assumption
http://scikit-learn.org/stable/modules/naive_bayes.html

"""

#from sklearn import datasets
#from sklearn.naive_bayes import GaussianNB
#iris = datasets.load_iris()
#gnb = GaussianNB()
#y_pred = gnb.fit(iris.data);

from sklearn.feature_extraction.text import TfidfVectorizer
import os


from util import log

raw_data_path = os.path.join(os.getcwd(),'data/raw_data/ex6DataEmails/nonspam-test/')
raw_data_file_list = [os.path.join(raw_data_path,fc) for fc in [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path,f))]]
for fc in raw_data_file_list:
	log.print_log(fc)


tfidf = TfidfVectorizer(raw_data_file_list, )
