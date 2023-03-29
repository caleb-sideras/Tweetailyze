import os
import openai
from celery import Celery
from dotenv import load_dotenv
from analysis_helper import preprocess_text, generate_embeddings, cluster_tweets, separate_clusters, create_output

load_dotenv(".env")

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND")

@celery.task(name="tweet_analysis")
def tweet_analysis(tweetsRaw):
    # Preprocess the text of each tweet using preprocess_text function
    tweets = [preprocess_text(t['text']) for t in tweetsRaw]

    # Get the tweet id for each tweet
    tweet_id = [t['id'] for t in tweetsRaw]

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    # Generate embeddings for each tweet using generate_embeddings function
    embeddings = generate_embeddings(tweets, openai)

    # Cluster tweets into 5 clusters using cluster_tweets function -> ideally some process should decide this
    cluster_labels = cluster_tweets(embeddings, 5)

    # Separate tweets into clusters based on their cluster labels using separate_clusters function
    clusters, clusters_id = separate_clusters(tweets, tweet_id, cluster_labels)

    # Create an output dictionary with key words, tweets, sentiment and topic weight for each cluster using create_output function
    out = create_output(clusters, clusters_id, 1, openai)

    return out

