install:
	pip install --upgrade pip &&\
		pip install -r app/requirements.txt

format:	
	black app/*.py 

lint:
	pylint --disable=R,C app/*.py

test:
	python -m pytest tests/ -vv --cov=app

hf-login:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF)

push-hub:
	huggingface-cli upload roeyzalta/every-right-rag-solution ./app/app.py --repo-type=space --commit-message="Update app.py"
	huggingface-cli upload roeyzalta/every-right-rag-solution ./app/requirements.txt --repo-type=space --commit-message="Update requirements"

deploy: hf-login push-hub

run:
	python app/app.py

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add .
	git commit -m "Update with new changes"
	git push origin $(BRANCH)

all: install format lint test update-branch deploy

.PHONY: install format lint test hf-login push-hub deploy run update-branch all