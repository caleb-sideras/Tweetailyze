from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from session import get_sessionmaker_instance
import uvicorn

from models import TwitterAccount, TweetCluster
import os

from celery_worker import tweet_analysis

import tweepy
from tweepy import TweepError
from twitter_helpers import get_twitter_user, get_tweets_by_user_id

app = FastAPI()

# Add the CORS middleware with allowed origins
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv(".env")

BEARER_TOKEN = os.environ["BEARER_TOKEN"]

SessionMaker = get_sessionmaker_instance()
session = SessionMaker()

def format_return(user, tweet_clusters):
    return {"user": user["data"], "clusters": tweet_clusters}

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
    try:
        user = get_twitter_user(tweepyClient=tweepyClient, username=username)
    except TweepError as e:
        return JSONResponse({"Error":str(e)}, status_code=400)

    # If the user is invalid, return an error response
    if not user:
        return JSONResponse({"Error":"Invalid Username"}, status_code=400)

    # Get the user ID from the user object
    user_id = user['data']['id']

    # TODO come up with some logic that only updates if most recent tweet > date modified
    # pulling a single most recent tweet?
    twitter_account = session.query(TwitterAccount).filter_by(id=user_id).first()

    if twitter_account:
        clusters = session.query(TweetCluster).filter(TweetCluster.account_id == user_id).all()
        tweet_clusters = []
        for cluster in clusters:
            cluster_dict = {
                "topic": cluster.topic,
                "key_words": cluster.key_words,
                "tweets": []
            }
            cluster_dict["tweets"].extend(map(lambda tweet: {"tweet_id": str(tweet.id), **tweet.data}, cluster.tweets))
            tweet_clusters.append(cluster_dict)
    
        return format_return(user, tweet_clusters)

    twitter_account = TwitterAccount(id=user_id, username=username)
    session.add(twitter_account)
    session.commit()

    # Retrieve the user's tweets using the Tweepy API client
    try:
        tweets = get_tweets_by_user_id(tweepyClient, user_id=user_id, max_results=20, expansions='attachments.media_keys', media_fields='url')
    except TweepError as e:
        return JSONResponse({"Error":str(e)}, status_code=400)
    
    # If the tweets are invalid, return an error response
    if not tweets:
        return JSONResponse({"Error":"Invalid User_Id"}, status_code=400)

    # Analyze the user's tweets using a Celery task
    task = tweet_analysis.delay(tweets['data'], user_id)

    # Get the results of the Celery task
    new_data = task.get()

    # Return the user's tweets and analysis results as a response
    return JSONResponse({format_return(user, new_data)})

# to run app locally, (keep docker running db)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
