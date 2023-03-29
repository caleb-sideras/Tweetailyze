# databae init 
docker-compose run app alembic revision --autogenerate -m "New Migration"
docker-compose run app alembic upgrade head

# looks at docker file and runs commands (requirements.txt etc)
docker-compose build

# looks at compose: downloads images, envs etc
docker-compose up