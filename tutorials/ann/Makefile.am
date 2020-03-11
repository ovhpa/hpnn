## these are now "tutorial" software

AM_CPPFLAGS = \
        -DPACKAGE_LOCALE_DIR=\""$(localedir)"\" \
        -DPACKAGE_SRC_DIR=\""$(srcdir)"\" \
        -DPACKAGE_DATA_DIR=\""$(pkgdatadir)"\" \
        $(LIBHPNN_CFLAGS)

AM_CFLAGS = -I$(top_srcdir)/include

bin_PROGRAMS = pdif

pdif_SOURCES = file_dif.c prepare_dif.c

pdif_LDADD = $(top_srcdir)/src/libhpnn.la


