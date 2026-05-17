import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(women_path, men_path):
    df_women = pd.read_csv(women_path)
    df_men = pd.read_csv(men_path)
    df_women['target'] = 'Women'
    df_men['target'] = 'Men'
    return pd.concat([df_women, df_men], ignore_index=True)

def preprocess_data(df):
    cols_to_drop = ['title', 'priceWithCurrency', 'availableText', 'lastUpdated', 'itemLocation']
    df = df.drop(columns=cols_to_drop)
    
    df['sold'] = df['sold'].fillna(0)
    df['available'] = df['available'].fillna(df['available'].median())
    
    df['type'] = df['type'].str.lower().str.strip().fillna('unknown')
    df['brand'] = df['brand'].str.lower().str.strip().fillna('unknown')
    
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    df['brand'] = le.fit_transform(df['brand'])
    df['target'] = df['target'].map({'Women': 0, 'Men': 1})
    
    return df

if __name__ == "__main__":
    w_path = "dataset_raw/ebay_womens_perfume.csv"
    m_path = "dataset_raw/ebay_mens_perfume.csv"
    
    print("Memulai data loading...")
    raw_df = load_data(w_path, m_path)
    
    print("Memulai proses preprocessing...")
    clean_df = preprocess_data(raw_df)

    output_path = "preprocessing/dataset_preprocessing.csv"
    
    clean_df.to_csv(output_path, index=False)
    print(f"Selesai! Data siap latih telah disimpan di: {output_path}")
# pancingan
