import pandas as pd

en_super_set_path = "/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/en_hf.csv"
# ucb_hatespeech_path =""

en_hatespeech_super_set = pd.read_csv(en_super_set_path)
# ucb_hatespeech = pd.read_csv(ucb_hatespeech_path)


super_set_neg_gender = pd.read_csv("/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/processed_data/en_hatespeech_superset/negative/gender_negative.csv")
super_set_neg_race = pd.read_csv("/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/processed_data/en_hatespeech_superset/negative/race_negative.csv")
super_set_neg_disability = pd.read_csv("/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/processed_data/en_hatespeech_superset/negative/disability_negative.csv")
super_set_neg_lgbt = pd.read_csv("/scratch/user/hasnat.md.abdullah/h2i_hatespeech_to_image/data/processed_data/en_hatespeech_superset/negative/lgbt_negative.csv")


