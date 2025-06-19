# Clean the dataset based on given rules
import pandas as pd
def clean_data(df):
  try:
      # 1. Keep only the specified columns
      required_columns = ['Index', 'News Headline', 'Complete News', 'Fake News(Yes/No)', 'word_count', 'character_count', 'Publish Dates', 'Source']
      if list(df.columns) != required_columns:
          print("Columns do not match the required list or order.")
          # 2. Remove rows with empty 'News Headline' or 'Complete News'
      df.dropna(subset=['News Headline', 'Complete News'], inplace=True)
      # Convert 'News Headline' to uppercase and 'Complete News' to lowercase
      df['News Headline'] = df['News Headline'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.upper()
      df['Complete News'] = df['Complete News'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
      # 3. Ensure 'Fake News(Yes/No)' contains only 1 or 0, replacing invalid values with 0
      df['Fake News(Yes/No)'] = df['Fake News(Yes/No)'].where(df['Fake News(Yes/No)'].isin([0, 1]), 1)
      # 4. Calculate missing 'word_count' and 'character_count'
      def count_words(text):
          return len(str(text).split())
  # Function to count characters excluding spaces in a text
      def count_characters(text):
          return len(str(text).replace(' ', ''))  # Removes spaces before counting
  # Apply the functions to the 'text_column'
      df['word_count'] = df['Complete News'].apply(count_words)
      df['character_count'] = df['Complete News'].apply(count_characters)
          # 5. Handle 'Publish Dates'
      # Drop rows where 'Publish Dates' is NaN (invalid or missing dates)
      df['Publish Dates'] = pd.to_datetime(df['Publish Dates'], format='mixed' , errors='coerce')    
      df.dropna(subset=['Publish Dates'], inplace = True)
      # 6. Handle 'Source' column
      df.loc[df['Source'].isna(), 'Fake News(Yes/No)'] = 1
      for item in df.index:
          if df.at[item, 'word_count']==0:
              df.drop(item, inplace=True)
      for item in df.index:
          if df.at[item, 'character_count']==0:
              df.drop(item, inplace=True)
  except:
      print("Falied to clean dataset, issue with csv file entered, please check and fix before uploading.")
  else:
      print("Dataset cleaned successfully.")
df=pd.read_csv('dataset/fake_news_final.csv')
clean_data(df)
