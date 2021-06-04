# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* yogAssit/*.py

black:
	@black scripts/* yogAssit/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr yogAssit-*.dist-info
	@rm -fr yogAssit.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

run_api:
	uvicorn api.fast_kpts:app_kpts --reload  # load web server with code autoreload

# ----------------------------------
#      GCP API Deploy
# ----------------------------------

# project id
PROJECT_ID=peppy-primacy-309809

# bucket name - replace with your GCP bucket name
DOCKER_IMAGE_NAME=yogassist
REGION=europe-west1
build_docker:
	-@docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

run_docker:
	docker run -e PORT=1234 -p 8000:1234 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

run_it:
	docker run -it -e PORT=1234 -p 8000:1234 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} sh

push_docker:
	-@docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

configure_api:
	gcloud config set project ${PROJECT_ID}

deploy_api:
	-@gcloud run deploy \
			--image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} \
			--platform managed \
			--region europe-west1 \
			--set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" \
			--memory '4Gi'

build_deploy:
	@make build_docker
	@make deploy_api
