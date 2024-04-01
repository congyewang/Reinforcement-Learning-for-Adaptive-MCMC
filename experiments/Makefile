SUBDIRS := $(wildcard trails/*/)

share_static: $(SUBDIRS)

default: share_static

$(SUBDIRS):
	$(MAKE) -C $@ share_static

.PHONY: share_static $(SUBDIRS)
