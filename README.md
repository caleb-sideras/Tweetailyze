<h1>TweetAIlyzeV1 - Twitter Summarizer</h1>
<p>This FastAPI backend provides a way to summarize a Twitter account through embedding and clustering of tweets, then performing sentiment analysis and topic modeling. Everything is commentend courtesy of ChatGPT.</p>
<h2>How it works</h2>
<ol>
  <li>The app takes a POST request of a Twitter account's username and gets the most recent X tweets using the tweepy package.</li>
  <li>The text of each tweet is preprocessed using the nltk package.</li>
  <li>The preprocessed text is transformed into embeddings using the transformers package and the pre-trained bert-base-multilingual-cased model.</li>
  <li>The embeddings are clustered using agglomerative clustering with cosine metric and average linkage.</li>
  <li>The tweets are separated into clusters based on their cluster labels.</li>
  <li>Sentiment analysis is performed using the VADER sentiment analyzer.</li>
  <li>Topic modeling is performed using non-negative matrix factorization (NMF) from the scikit-learn package.</li>
  <li>The top words for each topic are classified using the OpenAI completion API.</li>
  <li>The results are returned.</p>
</ol>
<h2>API Endpoint</h2>
<p>/tweets</p>
<h2>HTTP Method</h2>
<p>POST</p>
<h2>Request Body</h2>
<p>The request body should contain a JSON object with a single key-value pair where the key is "username" and the value is the users username. For example:</p>
<p>{"username":"elonmusk"}</p>
<h2>Response Body</h2>
<p>The response body is a JSON object with a single key-value pair where the key is Tweets and the value is another JSON object. The inner JSON object contains a key-value pair for each topic identified in the tweets. The key of each topic is a concatenated string of the most relevant Twitter handles and keywords associated with that topic. The value of each topic is another JSON object with two keys:</p>
<ul>
  <li>key_words: an array of strings representing the keywords associated with the topic</li>
  <li>tweets: an array of JSON objects representing the most relevant tweets associated with the topic. Each JSON object in the tweets array contains the following keys:
    <ul>
      <li>tweet: the text of the tweet</li>
      <li>tweet_id: the ID of the tweet</li>
      <li>sentiment: a JSON object representing the sentiment analysis of the tweet. The sentiment analysis contains the following keys:
        <ul>
          <li>neg: a float representing the negativity score of the tweet</li>
          <li>neu: a float representing the neutrality score of the tweet</li>
          <li>pos: a float representing the positivity score of the tweet</li>
          <li>compound: a float representing the overall sentiment score of the tweet (ranges from -1 to 1)</li>
        </ul>
      </li>
      <li>topic_weight: a float representing the relevance of the tweet to the given topic</li>
    </ul>
  </li>
</ul>
<p>Here's an example of what the response JSON object might look like:</p>
<pre>
{
  "Tweets": {
    "Dogecoin": {
      "key_words": [
        "dogeofficialceo",
        "days",
        "elvis",
        "theprashanthcb",
        "apart",
        "civilization",
        "mars",
        "hear",
        "getting",
        "falls"
      ],
      "tweets": [
        {
          "tweet": "peterdiamandis eat donut every morning still alive",
          "tweet_id": "1640783261713702914",
          "sentiment": {
            "neg": 0,
            "neu": 0.698,
            "pos": 0.302,
            "compound": 0.3818
          },
          "topic_weight": 0.00820937220701521
        },
        {
          "tweet": "theprashanthcb dogeofficialceo hear getting mars civilization falls apart",
          "tweet_id": "1640628523739258881",
          "sentiment": {
            "neg": 0,
            "neu": 1,
            "pos": 0,
            "compound": 0
          },
          "topic_weight": 0.6833158363483436
        }]
    }
   }
}
</pre>

<h2>Docker, FastAPI, FastAPI_SQLalchemy, PG ADMIN, Celery, Flower, Redis</h2>

<h2>Msc</h2>
<p>The only external api call is to OpenAI 'text-curie-001,' to classify the topic of a cluster based on key words. Embedding, Clustering, Topic modelling and sentiment analysis all run locally; reducing API fees at the cost of larger docker images and local server compute. TweetAIlyzeV2 will use OpenAI for embedding, removing the need for Tensorflow and Torch - significantly reducing the docker image sizes.</p>

<h2>Environment Variables (Examples)</h2>

<pre>
DATABASE_URL=postgresql+psycopg2://postgres:password@db:5432/example_db
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=example_db

PGADMIN_EMAIL=pgadmin@example.com
PGADMIN_PASSWORD=password

CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

OPENAI_API_KEY=

TWEEPY_BEARER_TOKEN=
</pre>
