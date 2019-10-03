# (Simple) Makefile
PREFIX=/usr/local
CURR_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

VPATH = src:mod_src:obj:modules

#include arch/make.gcc.p_openblas
#include arch/make.gcc.openblas
#include arch/make.icc.mkl
#include arch/make.icc.p_mkl
include arch/make.gcc.cuda

OBJ = $(CUDAOBJ) ann.o nn.o file.o
GOBJ = ann.og nn.og file.og
OBJ_DBG = $(CUDAOBJDBG) ann.obj nn.obj file.obj
GOBJ_DBG = ann.objg nn.objg file.objg


all: train_nn run_ann
debug: train_nn_dbg run_ann_dbg
gall: gtrain_nn grun_ann
gdebug: gtrain_nn_dbg grun_ann_dbg
everything: all debug gall gdebug

train_nn: $(OBJ)
	$(CC) $(INCFLAGS) $(CFLAGS) $(OPTFLAGS) ./train_nn.c -o ./train_nn $(OBJ) $(LDFLAGS) $(LIBS)
gtrain_nn: $(GOBJ)
	$(CC) $(INCFLAGS) $(CFLAGS) $(OPTFLAGS) $(GCFLAGS) ./train_nn.c -o ./gtrain_nn $(GOBJ) $(LDFLAGS) $(LIBS) $(GLFLAGS)
train_nn_dbg: $(OBJ_DBG)
	$(CC) $(INCDBGFLAGS) $(CFLAGS) $(DBGFLAGS) ./train_nn.c -o ./train_nn_dbg $(OBJ_DBG) $(DBGLDFLAGS) $(LIBS)
gtrain_nn_dbg: $(GOBJ_DBG)
	$(CC) $(INCDBGFLAGS) $(CFLAGS) $(DBGFLAGS) $(GCFLAGS) ./train_nn.c -o ./gtrain_nn_dbg $(GOBJ_DBG) $(DBGLDFLAGS) $(LIBS) $(GLFLAGS)
run_ann: $(OBJ)
	$(CC) $(INCFLAGS) $(CFLAGS) $(OPTFLAGS) ./run_nn.c -o ./run_nn $(OBJ) $(LDFLAGS) $(LIBS)
grun_ann: $(GOBJ)
	$(CC) $(INCFLAGS) $(CFLAGS) $(OPTFLAGS) $(GCFLAGS) ./run_nn.c -o ./grun_nn $(GOBJ) $(LDFLAGS) $(LIBS) $(GLFLAGS)
run_ann_dbg: $(OBJ_DBG)
	$(CC) $(INCDBGFLAGS) $(CFLAGS) $(DBGFLAGS) ./run_nn.c -o ./run_nn_dbg $(OBJ_DBG) $(DBGLDFLAGS) $(LIBS)
grun_ann_dbg: $(GOBJ_DBG)
	$(CC) $(INCDBGFLAGS) $(CFLAGS) $(DBGFLAGS) $(GCFLAGS) ./run_nn.c -o ./grun_nn_dbg $(GOBJ_DBG) $(DBGLDFLAGS) $(LIBS) $(GLFLAGS)
pdif: file_dif.o
	$(CC) $(INCFLAGS) $(CFLAGS) $(OPTFLAGS) ./prepare_dif.c -o ./pdif file_dif.o $(LDFLAGS) $(LIBS)
gpdif: file_dif.og
	$(CC) $(INCFLAGS) $(CFLAGS) $(OPTFLAGS) $(GCFLAGS) ./prepare_dif.c -o ./gpdif file_dif.og $(LDFLAGS) $(LIBS) $(GLFLAGS)
pdif_dbg: file_dif.obj
	$(CC) $(INCDBGFLAGS) $(CFLAGS) $(DBGFLAGS) ./prepare_dif.c -o ./pdif_dbg file_dif.obj $(DBGLDFLAGS) $(LIBS)


.SUFFIXES:
.SUFFIXES:      .c .cu .cc .C .cpp .objg .obj .og .o .mod .dmod

.c.o :
	$(CC) $(INCFLAGS) -o $@ -c $(CFLAGS) $(OPTFLAGS) $<
.c.og :
	$(CC) $(INCFLAGS) $(GCFLAGS) -o $@ -c $(CFLAGS) $(OPTFLAGS) $<
.c.obj :
	$(CC) $(INCDBGFLAGS) -o $@ -c $(CFLAGS) $(DBGFLAGS) $<
.c.objg :
	$(CC) $(INCDBGFLAGS) $(GCFLAGS) -o $@ -c $(CFLAGS) $(DBGFLAGS) $<
.cu.o :
	$(NVCC) $(NVINC) -o $@ -c $(NVFLAGS) $<
.cu.obj :
	$(NVCC) $(NVINC) -o $@ -c $(NVDBGFLAGS) $<



.FORCE:

count:
	wc *.c *.h *.def *.bash

clean:
	rm -f ./train_nn
	rm -f ./run_nn
	rm -f ./train_nn_dbg
	rm -f ./run_nn_dbg
	rm -f ./gtrain_nn
	rm -f ./grun_nn
	rm -f ./gtrain_nn_dbg
	rm -f ./grun_nn_dbg
	rm -f *.o *.og *.obj *.objg
	rm -f pdif pdif_dbg gpdif

.PHONY: .FORCE
.PHONY: all
.PHONY: debug
.PHONY: count
.PHONY: clean
