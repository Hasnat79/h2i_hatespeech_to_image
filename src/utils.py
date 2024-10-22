import pandas as pd
import os

def filter_superset_data(data_frame,keywords,output_path,labels):
  
  label_df = data_frame[data_frame['labels'] == labels]
  filtered_label_df = label_df[label_df['text'].str.contains('|'.join(keywords), case=False, na=False)]
  #check if path exists
  if not os.path.exists(output_path):
   filtered_label_df.to_csv(output_path, index=False)

