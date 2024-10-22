from data import en_hatespeech_super_set
from utils import filter_superset_data

# gender_based_keywords
gender_keywords = ['gender', 'woman', 'man', 'female', 'male', 'girl', 'boy']#

# filter data , label = 1 means negative / hatefull
#output_path 
output_path = '../data/processed_data/en_hatespeech_superset_gender_negative.csv'
filter_superset_data(en_hatespeech_super_set, gender_keywords,output_path,labels=1)

#output_path
# filter data , label = 0 means positive / non-hatefull
output_path = '../data/processed_data/en_hatespeech_superset_gender_positive.csv'

filter_superset_data(en_hatespeech_super_set,gender_keywords,output_path,labels=0)
