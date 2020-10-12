
docs/index.html:
	pdoc --html\
	    --output-dir docs\
	    --config latex_math=True\
	    --config show_source_code=False\
	    --force aisa
	mv -f docs/aisa/*.html docs/
	rmdir docs/aisa
