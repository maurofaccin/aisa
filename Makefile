
all: docs/index.html build


build: dist/aisa-1.0.0-py3-none-any.whl dist/aisa-1.0.0.tar.gz


docs/index.html: README.md aisa/*.py
	pdoc --html\
	    --output-dir docs\
	    --config latex_math=True\
	    --config show_source_code=False\
	    --force aisa
	mv -f docs/aisa/*.html docs/
	rmdir docs/aisa


dist/aisa-1.0.0-py3-none-any.whl:
	python -m build --wheel -o dist --no-isolation


dist/aisa-1.0.0.tar.gz:
	python -m build --sdist -o dist --no-isolation


.PHONY: all build
