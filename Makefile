.PHONY: check lint status

check:
	python -m py_compile *.py

lint:
	ruff check .

status:
	git status --short --branch

