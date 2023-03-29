from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi_sqlalchemy import DBSessionMiddleware, db
from dotenv import load_dotenv

from schema import TwitterAccount as SchemaTwitterAccount
from models import TwitterAccount
import os

from celery_worker import tweet_analysis

import tweepy
from twitter_helpers import get_twitter_user, get_tweets_by_user_id

app = FastAPI()

load_dotenv(".env")

app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])
BEARER_TOKEN = os.environ["BEARER_TOKEN"]

@app.post("/tweets")
def check_username(data=Body(...)):

    # Retrieve the username from the request data
    username = str(data["username"])

    # Initialize a Tweepy API client with the bearer token
    tweepyClient = tweepy.Client(
        bearer_token=BEARER_TOKEN, 
        return_type=dict
    )

    # Get the Twitter user object using the Tweepy API client
    user = get_twitter_user(tweepyClient=tweepyClient, username=username)

    # If the user is invalid, return an error response
    if not user:
        return JSONResponse({"Error":"Invalid Username"})

    # Get the user ID from the user object
    user_id = user['data']['id']

    # Check if the user's tweets are already stored in the database
    twitter_account = db.session.query(TwitterAccount).filter_by(id=user_id).first()

    # If the tweets are already stored, return them as a response
    if twitter_account:
        return JSONResponse({"Tweets":twitter_account.data})

    # Retrieve the user's tweets using the Tweepy API client
    tweets = get_tweets_by_user_id(tweepyClient, user_id=user_id, max_results=20, expansions='attachments.media_keys', media_fields='url')

    # If the tweets are invalid, return an error response
    if not tweets:
        return JSONResponse({"Error":"Invalid User_Id"})

    # Analyze the user's tweets using a Celery task
    task = tweet_analysis.delay(tweets['data'])

    # Get the results of the Celery task
    new_data = task.get()

    # Store the user's tweets and analysis results in the database
    new_account = TwitterAccount(id=user_id, username=username, data=new_data)
    db.session.add(new_account)
    db.session.commit()

    # Return the user's tweets and analysis results as a response
    return JSONResponse({"Tweets":new_data})