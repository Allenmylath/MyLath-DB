# Makefile
.PHONY: install test lint format clean docs

install:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=mylath

lint:
	flake8 mylath
	mypy mylath

format:
	black mylath tests
	isort mylath tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

docs:
	cd docs && make html

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

redis-start:
	docker run -d --name mylath-redis -p 6379:6379 redis:7-alpine

redis-stop:
	docker stop mylath-redis
	docker rm mylath-redis
