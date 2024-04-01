SUBDIRS := $(wildcard trails/*/)

share: $(SUBDIRS)

default: share

$(SUBDIRS):
	$(MAKE) -C $@ share

.PHONY: share $(SUBDIRS)
