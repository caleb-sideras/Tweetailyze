from sqlalchemy import Column, DateTime, ForeignKey,String, JSON, ARRAY, Integer, Float
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class TwitterAccount(Base):
    __tablename__ = "twitteraccount"
    id = Column(String, primary_key=True, index=True, autoincrement=False)
    username = Column(String)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

class TweetCluster(Base):
    __tablename__ = "tweetcluster"
    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String)
    key_words = Column(ARRAY(String))

    account_id = Column(String, ForeignKey("twitteraccount.id"))
    account = relationship("TwitterAccount")
    tweets = relationship("Tweets", backref="tweet_cluster")

class Tweets(Base):
    __tablename__= "tweet"
    id = Column(String, primary_key=True, index=True, autoincrement=False)
    data = Column(JSON)

    cluster_id = Column(Integer, ForeignKey("tweetcluster.id"))
    cluster = relationship("TweetCluster")

  