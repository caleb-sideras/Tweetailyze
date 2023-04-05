# databae init 
docker-compose run app alembic revision --autogenerate -m "New Migration"
docker-compose run app alembic upgrade head

# looks at docker file and runs commands (requirements.txt etc)
docker-compose build

# looks at compose: downloads images, envs etc
docker-compose up


# PG AMDIN

 
pgadmin:

container_name: pgadmin
image: dpage/pgadmin4
environment:
    - PGADMIN_DEFAULT_EMAIL=${PGADMIN_EMAIL}
    - PGADMIN_DEFAULT_PASSWORD=${PGADMIN_PASSWORD}
ports:
    - 5050:80
depends_on:
    - db

# Flower
flower:
container_name: flower
build: .
command: celery -A celery_worker.celery flower --port=5555
ports:
    - 5556:5555
environment:
    - CELERY_BROKER_URL=${CELERY_BROKER_URL}
    - CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND}
depends_on:
    - app
    - redis
    - celery_worker

# Caddy
caddy:
    build: ./caddy
    image: calebsideras/caddy:2.6.4
    ports:
      - 443:443
      - 80:80
    privileged: true
    depends_on:
      - app



FROM caddy:2.6.4
COPY Caddyfile /etc/caddy/Caddyfile


ec2-3-138-120-20.us-east-2.compute.amazonaws.com {
  reverse_proxy localhost:8000
}