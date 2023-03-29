from sqlalchemy import Column, DateTime,String, JSON
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.sql import func

Base = declarative_base()

class TwitterAccount(Base):
    __tablename__ = "twitteraccount"
    id = Column(String, primary_key=True)
    username = Column(String)
    data = Column(JSON)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
  