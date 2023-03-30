
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

import os
from dotenv import load_dotenv
load_dotenv(".env")

"""
The function creates a new database engine object using the create_engine() function from SQLAlchemy, which takes a database URI as its argument. The pool_pre_ping parameter is set to True, which enables the engine to test database connections before using them.
Then the function creates and returns a regular sessionmaker instance that is also bound to the engine created earlier. The sessionmaker instance can be used to create new sessions whenever they are needed in the application.
"""

def get_sessionmaker_instance() -> sessionmaker:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session

# The scoped_session object is a session that provides a thread-local scope for a single SQLAlchemy Session object. 
# It automatically handles the opening and closing of sessions as needed by the current thread, and is typically used in a web application or other multi-threaded application.
    # db_session = scoped_session(
    #     sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # )