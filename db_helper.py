from schema import TweetCluster as SchemaTweetCluster
from schema import Tweets as SchemaTweets
from models import TweetCluster, Tweets

from celery import Celery
import os
from dotenv import load_dotenv

from session import get_sessionmaker_instance

load_dotenv(".env")

celery = Celery('tweet_analysis', broker=os.environ.get("CELERY_BROKER_URL"), backend=os.environ.get("CELERY_RESULT_BACKEND")) 

SessionMaker = get_sessionmaker_instance()

"""
The Session is automatically cleaned up when the block is exited, 
and any changes made to the database are committed. 
This ensures that database resources are used efficiently and that the data remains consistent.

When you retrieve an object from the database using a session, 
it becomes part of the session's managed state. 
This means that any changes you make to the object will be tracked by the session,
and persisted to the database when the session is committed. 

TODO Try and get the Schema type workings (for type checks)
"""
@celery.task
def db_cluster(cluster: dict):
    print("db_cluster")
    with SessionMaker() as session:
        db_cluster = TweetCluster(
            topic=cluster["topic"],
            key_words=cluster["key_words"],
            account_id=cluster["account_id"]
        )
        session.add(db_cluster)
        session.commit()

        # Code below **needed if returning the db_cluster object
        # Update an object with the latest state from the database
        session.refresh(db_cluster)
        # Detach the tweet_cluster object from the session (turns it into a python object)
        session.expunge(db_cluster)

        return db_cluster.id

@celery.task
def db_tweet(tweet: dict):
    print("db_tweet")
    with SessionMaker() as session:
        db_tweet = Tweets(
            id=tweet["id"],
            data=tweet["data"],
            cluster_id=tweet["cluster_id"],
        )
        session.add(db_tweet)
        session.commit()
        return db_tweet