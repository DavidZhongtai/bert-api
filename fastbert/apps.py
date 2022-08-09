from django.apps import AppConfig
import html
import pathlib
import os

from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np


class WebappConfig(AppConfig):
    name = 'fastbert'
    model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
    articles = pd.read_json('fastbert/model/process_w_abstract.json')
    index = faiss.read_index('fastbert/model/articles_abstract_only.index')

    def fetch_article_info(self, dataframe_idx, df):
        info = df.iloc[dataframe_idx]
        meta_dict = dict()
        meta_dict['title'] = info['title']
        meta_dict['description'] = info['description']
        meta_dict['doi'] = info['doi']
        meta_dict['author_count'] = info['author_count']
        meta_dict['author_names'] = info['author_names']
        meta_dict['authkeywords'] = info['authkeywords']
        return meta_dict

    def search(self, query, top_k, index, model, df):
        print(index.ntotal)
        query_vector = model.encode(query)
        top_k = index.search([query_vector], top_k)
        # top_k = faiss.IndexRefineFlat(top)

        # print('>>>> Results in Total Time: {}'.format(time.time() - t))
        top_k_ids = top_k[1].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        results = [self.fetch_article_info(idx, df) for idx in top_k_ids]
        return results
