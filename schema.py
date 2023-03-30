from pydantic import BaseModel, Json
from typing import List


class TwitterAccount(BaseModel):
    id: str
    username: int

    class Config:
        orm_mode=True

class TweetCluster(BaseModel):
    topic: str
    key_words: List[str]
    account_id: str

    class Config:
        orm_mode=True

class Tweets(BaseModel):
    id: str
    data: Json
    cluster_id: str

    class Config:
        orm_mode=True