from django.shortcuts import render
from .apps import WebappConfig

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import WebappConfig
import numpy as np


def fetch_article_info(dataframe_idx, df):
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
    top_k = index.search(query_vector, top_k)
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results = [self.fetch_article_info(idx, df) for idx in top_k_ids]
    return results


class call_model(APIView):
    def get(self, request):
        if request.method == 'GET':
            # sentence is the query we want to get the prediction for
            params = request.GET.get('query', 'default')

            # predict method used to get the prediction
            response = WebappConfig.model.encode([params])
            # response = search(self, query=params, top_k=15, index=WebappConfig.index, model=WebappConfig.model, df=WebappConfig.articles)
            # response = search(self, )
            response = WebappConfig.index.search(response, 15)
            # nice
            # returning JSON response
            return JsonResponse(response.tolist(), safe=False)
