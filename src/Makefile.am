## Process this file with automake to produce Makefile.in

## Created by Anjuta

AM_CPPFLAGS = \
	-DPACKAGE_LOCALE_DIR=\""$(localedir)"\" \
	-DPACKAGE_SRC_DIR=\""$(srcdir)"\" \
	-DPACKAGE_DATA_DIR=\""$(pkgdatadir)"\" \
	$(LIBHPNN_CFLAGS)

AM_CFLAGS =\
	 -Wall\
	 -g

lib_LTLIBRARIES = libhpnn.la


libhpnn_la_SOURCES = \
	lib.c

libhpnn_la_LDFLAGS = 

libhpnn_la_LIBADD = $(LIBHPNN_LIBS)



include_HEADERS = \
	libhpnn.h

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libhpnn-0.1.pc

EXTRA_DIST = \
	libhpnn-0.1.pc.in
