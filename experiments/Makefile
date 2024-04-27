SUBDIRS := $(wildcard trails/*/)

share_static: $(SUBDIRS)
post_build: $(SUBDIRS)

default: share_static post_build

$(SUBDIRS):
	$(MAKE) -C $@ share_static
	$(MAKE) -C $@ post_build

.PHONY: default share_static post_build $(SUBDIRS)
