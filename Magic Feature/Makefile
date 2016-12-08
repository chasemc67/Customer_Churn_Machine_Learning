
EXE := magic_feature.pyz
PYTHON := "/usr/bin/env python"
TDIR := $(shell mktemp -d)
TFILE := $(shell mktemp)

all: $(EXE)

$(EXE):
	python3 -m zipapp . -p $(PYTHON) -o $(TFILE)
	mv $(TFILE) $(EXE)
	rm -r $(TDIR)

debug:
	cp -r * $(TDIR)
	python3 -m zipapp $(TDIR) -p $(PYTHON) -o $(TFILE)
	mv $(TFILE) $(EXE)
	rm -r $(TDIR)

clean:
	rm $(EXE)
