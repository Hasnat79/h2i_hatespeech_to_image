import pandas as pd
#uncomment when using spider_man
# dir_path = "/home/grads/h/hasnat.md.abdullah/h2i_hatespeech_to_image/data/"
#uncomment when using hprc
dir_path = "/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data"
en_super_set_path = dir_path+"en_hf.csv"
# ucb_hatespeech_path =""

en_hatespeech_super_set = pd.read_csv(en_super_set_path)
# ucb_hatespeech = pd.read_csv(ucb_hatespeech_path)


super_set_non_hateful_gender = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/non_hateful/gender_non_hateful.csv")
super_set_non_hateful_race = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/non_hateful/race_non_hateful.csv")
super_set_non_hateful_disability = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/non_hateful/disability_non_hateful.csv")
super_set_non_hateful_lgbt = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/non_hateful/lgbt_non_hateful.csv")


super_set_hateful_gender = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/hateful/gender_hateful.csv")
super_set_hateful_race = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/hateful/race_hateful.csv")
super_set_hateful_disability = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/hateful/disability_hateful.csv")
super_set_hateful_lgbt = pd.read_csv(dir_path+"processed_data/en_hatespeech_superset/hateful/lgbt_hateful.csv")

