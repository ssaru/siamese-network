build:
	pipreqs ./ --force

test:
	pip3 install -r tests/requirements.txt
	pytest --pylint --mypy --doctest-modules

format:
	isort -y
	black icarebot

dep:
	pip3 install -r requirements.txt
