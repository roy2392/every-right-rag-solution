install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

lint:
	pylint --disable=R,C *.py

test:
	python -m pytest tests/ -vv --cov=.

hf-login:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF)

push-hub:
	huggingface-cli upload roeyzalta/every-right-rag-solution ./app.py --repo-type=space --commit-message="Update app.py"
	huggingface-cli upload roeyzalta/every-right-rag-solution ./requirements.txt --repo-type=space --commit-message="Update requirements"
	huggingface-cli upload roeyzalta/every-right-rag-solution ./.env.example --repo-type=space --commit-message="Update .env.example"

deploy: hf-login push-hub

run:
	python app.py

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add .
	git commit -m "Update with new changes"
	git push origin $(BRANCH)

all: install format lint test update-branch deploy

.PHONY: install format lint test hf-login push-hub deploy run update-branch all