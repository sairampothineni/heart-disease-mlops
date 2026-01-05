VENV=venv
PYTHON=py -3.11

all: venv install test

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV)/Scripts/python.exe -m pip install --upgrade pip
	$(VENV)/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.txt

test:
	$(VENV)/Scripts/python.exe -m pytest -v

clean:
	rm -rf $(VENV)
