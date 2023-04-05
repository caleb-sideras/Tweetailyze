from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from session import get_sessionmaker_instance
import uvicorn

from models import TwitterAccount, TweetCluster
import os

from celery_worker import tweet_analysis

import tweepy
from twitter_helpers import get_twitter_user, get_tweets_by_user_id

app = FastAPI()

# Add the CORS middleware with allowed origins
# origins = ["http://localhost:5173", "https://tweetailyze-frontend.vercel.app/", "https://tweetailyze-frontend.vercel.app"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

load_dotenv(".env")

BEARER_TOKEN = os.environ["BEARER_TOKEN"]

SessionMaker = get_sessionmaker_instance()
session = SessionMaker()

def format_return(user, tweet_clusters):
    return {"user": user["data"], "clusters": tweet_clusters}

def delete_account(twitter_account):
    session.delete(twitter_account)
    session.commit()

@app.get("/")
def default_endpoint():
    return({"hello":"this is i"})

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
    except Exception as e:
        return JSONResponse({"Error":str(e)}, status_code=400)

    # If the user is invalid, return an error response
    if not user or 'errors' in user or (not 'data' in user):
        print(user)
        return JSONResponse({"Error":"Invalid Username"}, status_code=400)
    
    # Get the user ID from the user object
    user_id = user['data']['id']

    # TODO come up with some logic that only updates if most recent tweet > date modified
    # pulling a single most recent tweet?
    try:
        twitter_account = session.query(TwitterAccount).filter_by(id=user_id).first()
    except Exception as e:
        return JSONResponse({"Error":str(e)}, status_code=400)

    if twitter_account:
        try:
            clusters = session.query(TweetCluster).filter(TweetCluster.account_id == user_id).all()
        except Exception as e:
            return JSONResponse({"Error":str(e)}, status_code=400)
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

    try:
        twitter_account = TwitterAccount(id=user_id, username=username)
        session.add(twitter_account)
        session.commit()
    except Exception as e:
        return JSONResponse({"Error":str(e)}, status_code=400)

    # Retrieve the user's tweets using the Tweepy API client
    try:
        tweets = get_tweets_by_user_id(tweepyClient, user_id=user_id, max_results=20)
    except Exception as e:
        delete_account(twitter_account)
        return JSONResponse({"Error":str(e)}, status_code=400)
    
    # If the tweets are invalid, return an error response
    if not tweets or 'errors'in tweets or (not 'data' in tweets):
        print(tweets)
        return JSONResponse({"Error":"Unknown Error"}, status_code=400)

    
    # Analyze the user's tweets using a Celery task
    task = tweet_analysis.delay(tweets['data'], user_id)

    # Get the results of the Celery task
    try:
        new_data = task.get()
    except Exception as e:
        delete_account(twitter_account)
        out = "Task failed with error: {}".format(task.traceback) if task.state == "FAILURE" else "Unknown error occurred: {}".format(str(e))
        return JSONResponse({out}, status_code=400)
    
    if not new_data:
        return JSONResponse({"Error":"Unknown Error"}, status_code=400)

    # Return the user's tweets and analysis results as a response
    return JSONResponse(format_return(user, new_data))

# to run app locally, (keep docker running db)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
