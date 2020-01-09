## these are now "test" software

AM_CPPFLAGS = \
        -DPACKAGE_LOCALE_DIR=\""$(localedir)"\" \
        -DPACKAGE_SRC_DIR=\""$(srcdir)"\" \
        -DPACKAGE_DATA_DIR=\""$(pkgdatadir)"\" \
        $(LIBHPNN_CFLAGS)

AM_CFLAGS = -I$(top_srcdir)/include

bin_PROGRAMS = run_nn train_nn

run_nn_SOURCES = run_nn.c
train_nn_SOURCES = train_nn.c 

run_nn_LDADD = $(top_srcdir)/src/libhpnn.la
train_nn_LDADD = $(top_srcdir)/src/libhpnn.la


