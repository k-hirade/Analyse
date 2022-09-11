from django.shortcuts import render
import pandas as pd
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AdamW
# --twitter読み込み-------------------------------------
from requests_oauthlib import OAuth1Session
import tweepy
import torch
import glob
import os

# jsonで送信
from django.http.response import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from .models import Data

# logging.debug('hi_debug')


def search_tweets(request):
    if request.method == 'POST':
        username = request.POST['username']
        # ログインユーザーによって取得する値を変更する為、後日下記情報DBから取得出来るように変更
        oauth_session_params = {}
        oauth_session_params['consumer_key'] = 'CegT2Ta2nj3d8kTJ6Zv6TZBRY'
        oauth_session_params['consumer_secret'] = 'HypTVfMZV0SvaL7fs5vIbBJ8IseumpV7y09uT9JKid34wpNBEh'
        oauth_session_params['access_token'] = '1370365289750089728-hOE685rMvNOawPfdHI4QngjlTXi6JZ'
        oauth_session_params['access_secret'] = 'McQvl3JkbZ35dKHqpdX6xBYDpOMcNc0wHkeZEzdOkHmGu'
        # --------------------------------------
        auth = tweepy.OAuthHandler(
            oauth_session_params['consumer_key'], oauth_session_params['consumer_secret'])
        auth.set_access_token(
            oauth_session_params['access_token'], oauth_session_params['access_secret'])
        api = tweepy.API(auth)
        twitterApi = TwitterApi(oauth_session_params)
        tweets = twitterApi.get_user_twieet(username, api)
    else:
        username = ''
        tweets = ''
    context = {
        'username': username,
        'tweets': tweets,
    }
    return render(request, 'analyse.html', context)

# logging.debug('hi_debug')


class TwitterApi:
    def __init__(self, oauth_session_params):
        CONSUMER_KEY = oauth_session_params['consumer_key']
        CONSUMER_SECRET = oauth_session_params['consumer_secret']
        ACCESS_TOKEN = oauth_session_params['access_token']
        ACCESS_SECRET = oauth_session_params['access_secret']
        self.twitter = OAuth1Session(
            CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
        self.url_trend = "https://api.twitter.com/1.1/trends/place.json"
        self.url_search = "https://api.twitter.com/1.1/search/tweets.json"
        self.url_embed = "https://publish.twitter.com/oembed"
        self.url_status = "https://api.twitter.com/1.1/statuses/show.json"

    def get_user_twieet(self, username, api):
        # ラベリングするためのデータフレーム
        data = pd.DataFrame(columns=['文章', 'label_強欲', 'score_強欲', 'label_嫉妬', 'score_嫉妬', 'label_怠惰',
                            'score_怠惰', 'label_憤怒', 'score_憤怒', 'label_暴食', 'score_暴食', 'label_傲慢', 'score_傲慢', 'label_色欲', 'score_色欲'])
        data_output = pd.DataFrame(columns=['文章', 'label_強欲', 'score_強欲', 'label_嫉妬', 'score_嫉妬', 'label_怠惰',
                                            'score_怠惰', 'label_憤怒', 'score_憤怒', 'label_暴食', 'score_暴食', 'label_傲慢', 'score_傲慢', 'label_色欲', 'score_色欲'])

        # 学習データ
        csv_file = pd.read_csv(glob.glob(
            '/Users/hiradekeishi/Desktop/k-hirade_private/demo/csv_data/ラベリングデータ.csv')[0], engine='python')
        # csv_file = pd.DataFrame(columns=['強欲', 'ラベル_強欲', '嫉妬', 'ラベル_嫉妬',
        #                         '怠惰', 'ラベル_怠惰', '憤怒', 'ラベル_憤怒', '暴食', 'ラベル_暴食', '傲慢', 'ラベル_傲慢', ])
        # 学習データのラベルをintタイプに
        csv_file["ラベル_強欲"] = csv_file["ラベル_強欲"].astype(float).astype(int)
        csv_file["ラベル_嫉妬"] = csv_file["ラベル_嫉妬"].astype(float).astype(int)
        csv_file["ラベル_怠惰"] = csv_file["ラベル_怠惰"].astype(float).astype(int)
        csv_file["ラベル_憤怒"] = csv_file["ラベル_憤怒"].astype(float).astype(int)
        csv_file["ラベル_暴食"] = csv_file["ラベル_暴食"].astype(float).astype(int)
        csv_file["ラベル_傲慢"] = csv_file["ラベル_傲慢"].astype(float).astype(int)
        csv_file["ラベル_色欲"] = csv_file["ラベル_色欲"].astype(float).astype(int)

        # ラベリングする本文を取得
        results = api.user_timeline(
            screen_name=username, count=300, tweet_mode='extended')
        for r in results:
            # r.textで、投稿の本文のみ取得する
            data = data.append(
                {'文章': r.full_text.replace("\n", "")}, ignore_index=True)
# 強欲について
        train_docs_greed = csv_file["強欲"].tolist()
        train_labels_greed = csv_file["ラベル_強欲"].tolist()
        output_docs_greed = data["文章"].tolist()

        # これを変えれば使うモデルが変わる？
        model_name = "cl-tohoku/bert-large-japanese"
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        encodings_greed = tokenizer(train_docs_greed, return_tensors='pt',
                                    padding=True, truncation=True, max_length=128)
        input_ids_greed = encodings_greed['input_ids']
        attention_mask_greed = encodings_greed['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_greed).unsqueeze(0)
        outputs_greed = model(
            input_ids_greed, attention_mask=attention_mask_greed, labels=labels)
        loss_greed = outputs_greed.loss
        loss_greed.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_greed:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'文章': x, 'label_強欲': label, 'score_強欲': score}, ignore_index=True)
# ここまで強欲
# 嫉妬について
        train_docs_jealousy = csv_file["嫉妬"].tolist()
        train_labels_jealousy = csv_file["ラベル_嫉妬"].tolist()
        output_docs_jealousy = data["文章"].tolist()

        encodings_jealousy = tokenizer(train_docs_jealousy, return_tensors='pt',
                                       padding=True, truncation=True, max_length=128)
        input_ids_jealousy = encodings_jealousy['input_ids']
        attention_mask_jealousy = encodings_jealousy['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_jealousy).unsqueeze(0)
        outputs_jealousy = model(
            input_ids_jealousy, attention_mask=attention_mask_jealousy, labels=labels)
        loss_jealousy = outputs_jealousy.loss
        loss_jealousy.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_jealousy:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'label_嫉妬': label, 'score_嫉妬': score}, ignore_index=True)
# ここまで嫉妬
# 怠惰について
        train_docs_laziness = csv_file["怠惰"].tolist()
        train_labels_laziness = csv_file["ラベル_怠惰"].tolist()
        output_docs_laziness = data["文章"].tolist()

        encodings_laziness = tokenizer(train_docs_laziness, return_tensors='pt',
                                       padding=True, truncation=True, max_length=128)
        input_ids_laziness = encodings_laziness['input_ids']
        attention_mask_laziness = encodings_laziness['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_laziness).unsqueeze(0)
        outputs_laziness = model(
            input_ids_laziness, attention_mask=attention_mask_laziness, labels=labels)
        loss_laziness = outputs_laziness.loss
        loss_laziness.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_laziness:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'label_怠惰': label, 'score_怠惰': score}, ignore_index=True)
# ここまで怠惰
# 憤怒について
        train_docs_anger = csv_file["憤怒"].tolist()
        train_labels_anger = csv_file["ラベル_憤怒"].tolist()
        output_docs_anger = data["文章"].tolist()

        encodings_anger = tokenizer(train_docs_anger, return_tensors='pt',
                                    padding=True, truncation=True, max_length=128)
        input_ids_anger = encodings_anger['input_ids']
        attention_mask_anger = encodings_anger['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_anger).unsqueeze(0)
        outputs_anger = model(
            input_ids_anger, attention_mask=attention_mask_anger, labels=labels)
        loss_anger = outputs_anger.loss
        loss_anger.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_anger:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'label_憤怒': label, 'score_憤怒': score}, ignore_index=True)
# ここまで憤怒
# 暴食について
        train_docs_surfeit = csv_file["暴食"].tolist()
        train_labels_surfeit = csv_file["ラベル_暴食"].tolist()
        output_docs_surfeit = data["文章"].tolist()

        encodings_surfeit = tokenizer(train_docs_surfeit, return_tensors='pt',
                                      padding=True, truncation=True, max_length=128)
        input_ids_surfeit = encodings_surfeit['input_ids']
        attention_mask_surfeit = encodings_surfeit['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_surfeit).unsqueeze(0)
        outputs_surfeit = model(
            input_ids_surfeit, attention_mask=attention_mask_surfeit, labels=labels)
        loss_surfeit = outputs_surfeit.loss
        loss_surfeit.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_surfeit:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'label_暴食': label, 'score_暴食': score}, ignore_index=True)
# ここまで暴食
# 傲慢について
        train_docs_arrogance = csv_file["傲慢"].tolist()
        train_labels_arrogance = csv_file["ラベル_傲慢"].tolist()
        output_docs_arrogance = data["文章"].tolist()

        encodings_arrogance = tokenizer(train_docs_arrogance, return_tensors='pt',
                                        padding=True, truncation=True, max_length=128)
        input_ids_arrogance = encodings_arrogance['input_ids']
        attention_mask_arrogance = encodings_arrogance['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_arrogance).unsqueeze(0)
        outputs_arrogance = model(
            input_ids_arrogance, attention_mask=attention_mask_arrogance, labels=labels)
        loss_arrogance = outputs_arrogance.loss
        loss_arrogance.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_arrogance:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'label_傲慢': label, 'score_傲慢': score}, ignore_index=True)
# ここまで傲慢
# 色欲について
        train_docs_lust = csv_file["色欲"].tolist()
        train_labels_lust = csv_file["ラベル_色欲"].tolist()
        output_docs_lust = data["文章"].tolist()

        encodings_lust = tokenizer(train_docs_lust, return_tensors='pt',
                                   padding=True, truncation=True, max_length=128)
        input_ids_lust = encodings_lust['input_ids']
        attention_mask_lust = encodings_lust['attention_mask']

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer = AdamW(model.parameters(), lr=1e-5)
        labels = torch.tensor(train_labels_lust).unsqueeze(0)
        outputs_lust = model(
            input_ids_lust, attention_mask=attention_mask_lust, labels=labels)
        loss_lust = outputs_lust.loss
        loss_lust.backward()
        optimizer.step()

        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model, tokenizer=model_name)

        sentiment_analyzer("これは、テストのための文章です")

        db = pd.DataFrame(columns=['label', 'score'])
        for x in output_docs_lust:
            db = sentiment_analyzer(x)
            label = db[0]["label"]
            score = db[0]["score"]
            # data.append(sentiment_analyzer(x))
            data_output = data_output.append(
                {'label_色欲': label, 'score_色欲': score}, ignore_index=True)
# ここまで色欲
        data_output = data_output.to_html()
        return data_output


class PushDataList(LoginRequiredMixin, TemplateView):
    login_url = '/accounts/login/'

    def index(request):
        return render(request, 'analyse.html', {})
    # POSTメソッドでリクエストされたら実行するメソッド

    def post(self, request, *args, **kwargs):
        push_type = request.POST.get('push_type')
        response_body = {}
        if push_type == 'datalist':
            # ユーザーとツイートのネガポジをDBに保存する
            data_user = Data(user=request.user)
            data_user.save()
            data_user_tweeit = Data(user=request.user.datalist)
            data_user_tweeit.save()
            tweeit_data = int(request.POST.get('search_twieet'))
            response_body = {
                'result': 'success',
                'datalist': tweeit_data
            }
        if not response_body:
            response_body = {
                'result': 'already_exists'
            }
        return JsonResponse(response_body)
