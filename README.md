# Chatbot Backend

> Deceptive LLM Chatbot Backend for MLSA's Lost Messages Event in KIITFEST 

## Usage

- Install [Docker](https://docs.docker.com/engine/install/) on the machine according to platform and architecture.

- Create `.env` file in the respective project directory.

- Replace the subdomain in `Caddyfile` before deploying.

- Copy the `docker-compose.yml` and run the following command.

```bash
docker compose up -d
```
- With `docker run` use the below command.


```bash
docker run -p 8000:8000 -d --env-file .env chatbot-backend:latest
```
