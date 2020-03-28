## Process this file with automake to produce Makefile.in
## Re-written OVHPA @2019

SUBDIRS = src tests tutorials/ann tutorials/mnist

dist_doc_DATA = \
	README \
	COPYING \
	AUTHORS \
	ChangeLog \
	INSTALL \
	NEWS

AM_CFLAGS = -I$(top_srcdir)/include


# Remove doc directory on uninstall
uninstall-local:
	-rm -r $(docdir)
