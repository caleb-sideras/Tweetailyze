import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
stop_words = stopwords.words('english')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Define a function to preprocess the text of each tweet
def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word.lower() not in stop_words]

    # Remove non-alphabetic characters and convert to lowercase
    words = [word.lower() for word in words if word.isalpha()]

    # Join the words back into a string
    preprocessed_text = " ".join(words)

    return preprocessed_text

# Define a function to generate embeddings for a list of sentences using BERT model
def generate_embeddings(sentences):
    # Load the tokenizer and model from pretrained BERT model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = AutoModel.from_pretrained('bert-base-multilingual-cased')

    # Tokenize the sentences with padding and truncation
    tokenized = tokenizer(sentences, padding=True,
                          truncation=True, return_tensors='pt')

    # Get the embeddings from the last hidden layer of the model without gradient calculation
    with torch.no_grad():
        outputs = model(**tokenized)

    embeddings = outputs.last_hidden_state

    # Convert the embeddings to numpy array and reshape it to 2D matrix
    np_embeddings = embeddings.numpy()
    flat_embeddings = np_embeddings.reshape(np_embeddings.shape[0], -1)

    return flat_embeddings

# Define a function to cluster tweets using agglomerative clustering with cosine metric and average linkage
def cluster_tweets(embeddings, n_clusters):
    # Create an agglomerative clustering object with n_clusters and cosine metric and average linkage parameters
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric='cosine', linkage='average')

    # Fit the clustering object on the embeddings and get the cluster labels for each tweet
    cluster_labels = clustering.fit_predict(embeddings)

    return cluster_labels

# Define a function to separate tweets into clusters based on their cluster labels
def separate_clusters(tweets, tweet_id, cluster_labels):
    # Initialize an empty list to store the clusters and their ids
    clusters = []
    clusters_id = []

    # Loop through each cluster label from 0 to max label value
    for i in range(cluster_labels.max() + 1):
        # Initialize an empty list to store the tweets and ids for each cluster label
        cluster = []
        cluster_id = []

        # Loop through each tweet and its index in the tweets list
        for j, tweet in enumerate(tweets):
            # If the cluster label of the tweet matches the current cluster label, append the tweet and its id to the cluster list
            if cluster_labels[j] == i:
                cluster.append(tweet)
                cluster_id.append(tweet_id[j])

        # Append the cluster list and its id list to the clusters list
        clusters.append(cluster)
        clusters_id.append(cluster_id)

    return clusters, clusters_id

# Define a function to create a VADER sentiment analyzer object and analyze sentiment for each tweet in a cluster
def analyze_sentiment(cluster):

    # Create a VADER sentiment analyzer object
    analyzer = SentimentIntensityAnalyzer()

    # Initialize an empty list to store the sentiment scores for each tweet
    sentiments = []

    # Loop through each tweet in the cluster
    for tweet in cluster:
        # Analyze sentiment using VADER and get the polarity scores
        scores = analyzer.polarity_scores(tweet)

        # Append the scores to the sentiments list
        sentiments.append(scores)

    return sentiments

# Define a function to perform NMF topic modeling on a cluster of tweets and get the top words for each topic
def get_topics(cluster, num_topics):

    # Create TF-IDF vectorizer with stop words removal
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Transform the cluster into TF-IDF matrix
    tfidf = vectorizer.fit_transform(cluster)

    # Perform NMF topic modeling with num_topics and random state parameters
    nmf = NMF(n_components=num_topics, random_state=1,
              init='nndsvda', solver='mu', max_iter=300)  # tol=0.01
    nmf.fit(tfidf)

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Get the top 10 words for each topic by sorting the nmf components
    top_n = 10
    top_word_indices = np.argsort(nmf.components_)[:, :-top_n-1:-1]

    # Get the top words as a list of strings
    top_words = np.array(feature_names)[top_word_indices]
    top_words = top_words.tolist()[0]

    return top_words

# Define a function to classify the topic of a cluster using OpenAI completion API
def classify_topic(top_words, openai_api):

    # Join the top words into a comma-separated string
    post_top_words = ", ".join("'" + item + "'" for item in top_words)

    # Create a prompt for OpenAI completion API to classify the topic
    prompt = f"Classify the topic: {post_top_words}\nLabel:"

    # Call the OpenAI completion API with text-curie-001 model and zero temperature and other parameters
    response = openai_api.Completion.create(
        model="text-curie-001",
        prompt=prompt,
        temperature=0,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Get the label from the response
    label = (response["choices"][0]["text"]).replace("\n", "")

    return label

# Define a function to get the topic weights for each tweet in a cluster using NMF
def get_topic_weights(cluster, num_topics):

    # Create TF-IDF vectorizer with stop words removal
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # Transform the cluster into TF-IDF matrix
    tfidf = vectorizer.fit_transform(cluster)

    # Perform NMF topic modeling with num_topics and random state parameters
    nmf = NMF(n_components=num_topics, random_state=1,
              init='nndsvda', solver='mu', max_iter=300)  # tol=0.01
    nmf.fit(tfidf)

    # Get the topic weights for each tweet using nmf transform
    tweet_topics = nmf.transform(tfidf)

    return tweet_topics

# Define a function to create an output dictionary with key words, tweets, sentiment and topic weight for each cluster
def create_output(clusters, clusters_id, num_topics, openai_api):

    # Initialize an empty dictionary to store the output
    out = {}

    # Loop through each cluster and its index in the clusters list
    for i, cluster in enumerate(clusters):

        # Get the top words for each cluster using get_topics function
        top_words = get_topics(cluster, num_topics)

        # Classify the topic of each cluster using classify_topic function
        key = classify_topic(top_words, openai_api)

        # Initialize an empty dictionary for each key in the output dictionary
        out[key] = {}

        # Store the key words for each key in the output dictionary
        out[key]["key_words"] = top_words

        # Get the topic weights for each tweet in the cluster using get_topic_weights function
        tweet_topics = get_topic_weights(cluster, num_topics)

        # Initialize an empty list to store the tweets for each key in the output dictionary
        out[key]["tweets"] = []

        # Analyze sentiment for each tweet in the cluster using analyze_sentiment function
        sentiments = analyze_sentiment(cluster)

        # Loop through each tweet and its index in the cluster
        for j, tweet in enumerate(cluster):

            # Get the sentiment score for the tweet
            sentiment = sentiments[j]

            # Get the topic weight for the tweet
            topic_weight = tweet_topics[j].tolist()[0]

        # Append a dictionary with tweet, tweet_id, sentiment and topic_weight to the tweets list in the output dictionary
            out[key]["tweets"].append({
                "tweet": tweet,
                "tweet_id": clusters_id[i][j],
                "sentiment": sentiment,
                "topic_weight": topic_weight
            })

    return out
