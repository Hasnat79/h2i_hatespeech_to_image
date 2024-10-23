from data import en_hatespeech_super_set
from utils import filter_superset_data




def generate_superset_hatespeech_data (en_hatespeech_super_set, keywords, output_path):
  
  
  # if labels = 1, hateful/negative
  path = output_path.replace(".csv", "_negative.csv")
  filter_superset_data(en_hatespeech_super_set, keywords,path,labels=0)
  # label ==0 , positive
  path = output_path.replace(".csv","_positive.csv")
  filter_superset_data(en_hatespeech_super_set,keywords, path, labels=1)




# generating en hatespeech superset: gender specific
print(f"filtering gender specific data")
gender_keywords = ['bitch','slut','feminazi','stay in the kitchen', 'man up', 'gold digger', 'women are property','catfight','misandrist','alpha male']#
output_path = '../data/processed_data/en_hatespeech_superset/gender.csv'
generate_superset_hatespeech_data(en_hatespeech_super_set, gender_keywords, output_path)

# generating en_hatespeech_superset: disability specific
print(f"filtering disability specific data")
disability_keywords = [
        " retard",
        "retarded",
        "cripple",
        "crippled",
        "burden on society",
        "worthless",
        "crazy",
        "spaz",
        "spastic",
        "freak",
        "vegetable ",
        "invalid",
        "better off dead",
        "gimp"]
output_path = '../data/processed_data/en_hatespeech_superset/disability.csv'
generate_superset_hatespeech_data(en_hatespeech_super_set, disability_keywords, output_path)

# generating en_hatespeech_superset: race specific
print(f"filtering race specific data")
race_keywords = [
"nigger",
"monkey" ,
"go back to your country",
"illegal" ,
"terrorist" ,
"chink",
"spic",
"thug" ,
"white supremacy",
"genetic inferior"
]
output_path = '../data/processed_data/en_hatespeech_superset/race.csv'
generate_superset_hatespeech_data(en_hatespeech_super_set, race_keywords,output_path)

# generating en_hatespeech_superset: lgbt specific
print(f"filtering lgbt specific data")
lgbt_keywords = [
  "faggot" ,
"dyke",
"tranny",
"that's so gay" ,
"it’s just a phase" ,
"sodomite"
"no homo"
"groomer" ,
"burn in hell" ,
"agenda" 
]

output_path = '../data/processed_data/en_hatespeech_superset/lgbt.csv'
generate_superset_hatespeech_data(en_hatespeech_super_set, lgbt_keywords,output_path)

