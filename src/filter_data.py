from data import en_hatespeech_super_set
from utils import filter_superset_data

# gender_based_keywords
# gender_keywords = ['gender', 'woman', 'man', 'female', 'male', 'girl', 'boy']#
# """bitch"" 
# ""slut""
# ""feminazi""
# ""stay in the kitchen"" 
# ""man up"" (when used to enforce toxic masculinity)
# ""gold digger""
# ""women are property""
# ""catfight"" (used to trivialize conflict between women)
# ""misandrist"" (though often used to criticize feminism)
# ""alpha male"" (when used to promote gender hierarch"


gender_keywords = ['bitch','slut','feminazi','stay in the kitchen', 'man up', 'gold digger', 'women are property','catfight','misandrist','alpha male']#

# filter data , label = 1 means negative / hatefull
#output_path 
output_path = '../data/processed_data/en_hatespeech_superset_gender_negative_v2.csv'
filter_superset_data(en_hatespeech_super_set, gender_keywords,output_path,labels=1)

#output_path
# filter data , label = 0 means positive / non-hatefull
output_path = '../data/processed_data/en_hatespeech_superset_gender_positive_v2.csv'

filter_superset_data(en_hatespeech_super_set,gender_keywords,output_path,labels=0)
