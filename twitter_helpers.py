import tweepy

def get_twitter_user(tweepyClient: tweepy.API, ID: str = None, username: str = None):
    if (ID is None and username is None) or (ID is not None and username is not None):
        raise TypeError("Either ID or username must be provided, but not both.")

    try:
        if ID is not None:
            user = tweepyClient.get_user(ID)
        else:
            user = tweepyClient.get_user(username=username)
    except tweepy.errors.BadRequest as e:
        print(f"Error: {e}")
        return None
    
    return user


def get_tweets_by_user_id(tweepyClient: tweepy.API, user_id: str, max_results: int = 20, expansions: str = 'attachments.media_keys', media_fields: str = 'url') -> list:
    try:
        tweets = tweepyClient.get_users_tweets(id=user_id, max_results=max_results, expansions=expansions, media_fields=media_fields)
        return tweets
    except tweepy.errors.TweepError as e:
        print(f"Error: {e}")
        return None
