"""

Naive Bayes Classification
Supervised Learning - Classification technique
What is does and how - https://www.youtube.com/watch?v=EGKeC2S44Rs
Maths behind - Very Simple Probability - Bayes Theorem with 'naive' assumption
http://scikit-learn.org/stable/modules/naive_bayes.html
tfidf - Feature Extraction Technique

"""

from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.naive_bayes import MultinomialNB #96.9
#from sklearn.naive_bayes import GaussianNB 95.7
#from  sklearn.naive_bayes import BernoulliNB 95.7
import pandas as pd
from sklearn import metrics

from util import log

log.print_log('FEATURE EXTRACTION, TRANSFORMATION, TRAINING AND PREDICTION')
log.print_log('Getting Data ready for reading..Training Data')
raw_data_path = os.path.join(os.getcwd(),'data/raw_data/ex6DataEmails/nonspam-train/')
raw_data_file_list_ham = [os.path.join(raw_data_path,fc) for fc in [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path,f))]]
raw_data_path = os.path.join(os.getcwd(),'data/raw_data/ex6DataEmails/spam-train/')
raw_data_file_list_spam = [os.path.join(raw_data_path,fc) for fc in [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path,f))]]

training_X = []
training_y = []

for f in raw_data_file_list_ham:
	with open(f,'r') as file:
		training_X.append(file.read())
		training_y.append(0)

for f in raw_data_file_list_spam:
	with open(f,'r') as file:
		training_X.append(file.read())
		training_y.append(1)

log.print_log('Reading Done for Training Data')
log.print_log('Extracting Features, Fit and transform ( Making data ready for training)')
tfidf = TfidfVectorizer()
X_trained_matrix_sparse = tfidf.fit_transform(training_X) # Always need one Dimensional
log.print_log('Extracting Features, Fit and transform ( Making data ready for training) finished')
"""
Visibility code
features = tfidf.get_feature_names()
X_trained_matrix_dense = X_trained_matrix_sparse.toarray()
log.print_log('Features Extraction, Data ready for training Models')
log.print_log('A glimpse of data. You might not understand it')
tdf = pd.DataFrame(data=X_trained_matrix_dense, columns=features)
print tdf.head(10)
"""
log.print_log('***********************|||||||**********************')
log.print_log("TRAINING MODEL")
log.print_log('Using Naive Bayes Classification. - Multinomial Naive Bayes')
classifier_mnb = MultinomialNB()
log.print_log('Training Model with Sparse Matrix and extracted Features')
classifier_mnb.fit(X_trained_matrix_sparse, training_y)
log.print_log('Model trained. Ready for prediction...')
log.print_log('***********************|||||||**********************')

log.print_log('Getting Data ready for reading..This is Testing Data')
raw_data_path = os.path.join(os.getcwd(), 'data/raw_data/ex6DataEmails/nonspam-test/')
raw_data_file_list_ham = [os.path.join(raw_data_path, fc) for fc in [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path,f))]]
raw_data_path = os.path.join(os.getcwd(), 'data/raw_data/ex6DataEmails/spam-test/')
raw_data_file_list_spam = [os.path.join(raw_data_path, fc) for fc in [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path,f))]]
test_X = []
test_y = []

for f in raw_data_file_list_ham:
	with open(f,'r') as file:
		test_X.append(file.read())
		test_y.append(0)

for f in raw_data_file_list_spam:
	with open(f,'r') as file:
		test_X.append(file.read())
		test_y.append(1)

log.print_log('Reading Done of testing Data')
log.print_log('Extracting Features and Fit( Making data ready for testing)')
X_test_matrix_sparse = tfidf.transform(test_X)
log.print_log('Data ready for testing with trained Model')
log.print_log('***********************|||||||**********************')

log.print_log('PREDICTION WITH TRAINED MODEL')
y_predict_class = classifier_mnb.predict(X_test_matrix_sparse)
log.print_log('Prediction Done')
log.print_log('***********************|||||||**********************')
log.print_log('Checking Accuracy')
accuracy = metrics.accuracy_score(test_y,y_predict_class)
log.print_log('Accuracy is - ' + str(accuracy*100)+'%')
confusion = metrics.confusion_matrix(test_y,y_predict_class)
log.print_log('Confusion is - ' + str(confusion))
log.print_log('*********************||| END |||********************')




