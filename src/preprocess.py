import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertModel
import pickle
import dvc.api
from pathlib import Path
from src.helper import save_data

class Data_Preprocess(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")

    def handle_nan(self,df,target):
        for column in df.columns:
            if df[column].dtype=='object':
                if column==target:
                    df=df.loc[~df[target].isnull()]
                    print(f'Percentage of missing values in the target metric intent is: {(df[target].isnull().sum()/len(df)*100).round(2)}')
                else:
                    df[column].fillna('other',inplace=True)
            
            if df[column].dtype=='number':
                df[column].fillna(0,inplace=True)
    
        df['intent_id']=df['intent'].factorize()[0]       
        return df
    
    def id_to_intent(self,df):
         #ToDo: modify these lines
        intent_id_df = df[['intent', 'intent_id']].drop_duplicates().sort_values('intent_id')
        #intent_to_id = dict(intent_id_df.values)
        id_to_intent = dict(intent_id_df[['intent_id', 'intent']].values)
        save_data(id_to_intent,params['data']['utilities'],'id_to_intent.pkl')

    def feature_engineer(self,df):
        other=['US','AT','GH']
        df['geo_country'] = df['geo_country'].apply(lambda x: 'other' if x in other else x)
        df['device_type'] = df['device_type'].apply(lambda x: 'other' if x == 'Tablet' else x)
        df['browser_name'] = df['browser_name'].apply(lambda x: 'other' if x == 'Brave' else x)
        df.drop('date',axis=1,inplace=True)
        return df

    def get_labels(self,df):
        y=df['intent_id'] #labels
        df.drop('intent_id',axis=1,inplace=True)
        return df,y

    def split_data(self,X,y,test_size):
        df_train, df_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=test_size,random_state=42)
        return df_train, df_test, y_train, y_test
        
    def preprocess_data(self,df_train,df_test,y_train,y_test,cat_metrics):
        #Categorical data
        encoder = OneHotEncoder(handle_unknown = 'ignore')
        encoder.fit(df_train[cat_metrics])
        train_cat=encoder.transform(df_train[cat_metrics]).toarray()
        test_cat=encoder.transform(df_test[cat_metrics]).toarray()
        #Numerical data
        scaler = MinMaxScaler()
        values_train=df_train['daily_query_count'].values.reshape(-1,1)
        scaler.fit(values_train)
        train_num=scaler.transform(values_train)
        values_test=df_test['daily_query_count'].values.reshape(-1,1)
        test_num=scaler.transform(values_test)
        #Text data
        train_query=self.preprocess_text(df_train['search_query'])
        X_train=np.concatenate((train_cat,train_num,train_query),axis=1)

        test_query=self.preprocess_text(df_test['search_query'])
        X_test=np.concatenate((test_cat,test_num,test_query),axis=1)
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        data={
              'train.npy':X_train,
              'train_labels.npy':y_train,
              'test.npy': X_test,
              'test_labels.npy':y_test
             }
        return data

    def preprocess_text(self,queries):
        col_embed=[]
        for query in queries:
            col_embed.append(self.get_text_embeddings(query))
        #reduce the dimensionality of the embeddings
        #reduced_embed=dense(torch.tensor(col_embed))
        return np.array(col_embed)

    def get_text_embeddings(self,text):
        # Tokenize and convert text to embeddings
        input_ids = self.tokenizer.encode(text, return_tensors="pt",truncation=True)
        with torch.no_grad():
            outputs = self.model(input_ids)

        # The embeddings are contained in the last layer of the model's output
        last_hidden_states = outputs.last_hidden_state

        # You can extract the embedding for each token in the input
        # For a simple example, you can average the embeddings across all tokens
        average_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()

        return average_embedding



if __name__=="__main__":
    params = dvc.api.params_show()
    preprocess=Data_Preprocess()
    df=pd.read_csv(params["data"]["raw"])
    df=preprocess.handle_nan(df,target=params["process"]["target"])
    df=preprocess.feature_engineer(df)
    id_to_intent=preprocess.id_to_intent(df)
    X,y=preprocess.get_labels(df)
    df_train, df_test, y_train, y_test=preprocess.split_data(X,y,params["process"]["test_size"])
    cat_metrics=['market','geo_country','device_type','browser_name']
    data=preprocess.preprocess_data(df_train,df_test,y_train,y_test,cat_metrics)

    for name,data in data.items():
        save_data(data,params['data']['preprocessed'],name)



        