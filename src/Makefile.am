## Process this file with automake to produce Makefile.in
## Re-written OVHPA @2019

AM_CPPFLAGS = \
	-DPACKAGE_LOCALE_DIR=\""$(localedir)"\" \
	-DPACKAGE_SRC_DIR=\""$(srcdir)"\" \
	-DPACKAGE_DATA_DIR=\""$(pkgdatadir)"\" \
	$(LIBHPNN_CFLAGS)

AM_CFLAGS = -I$(top_srcdir)/include

lib_LTLIBRARIES = libhpnn.la

include_HEADERS = $(top_srcdir)/include/libhpnn.h

otherincludedir = $(includedir)/libhpnn
otherinclude_HEADERS = $(top_srcdir)/include/libhpnn/common.h \
	$(top_srcdir)/include/libhpnn/unroll.def \
	$(top_srcdir)/include/libhpnn/ann.h \
	$(top_srcdir)/include/libhpnn/cuda_ann.h

libhpnn_la_SOURCES = \
	libhpnn.c ann.c

if HAVE_CUDA
libhpnn_la_SOURCES += cuda_ann.cu

libhpnn_la_AR = $(NVCC) -lib -o

endif

libhpnn_la_LDFLAGS = 

libhpnn_la_LIBADD = $(LIBHPNN_LIBS)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libhpnn.pc

EXTRA_DIST = \
	libhpnn.pc.in

if HAVE_CUDA
# THE only working so far
.cu.lo:
	$(LIBTOOL) --tag=CC --mode=compile $(NVCC) --compile -Xcompiler="-fPIC -DPIC" $(NVCCFLAGS) $(AM_CFLAGS) -o $@ $^ -prefer-non-pic 
endif
