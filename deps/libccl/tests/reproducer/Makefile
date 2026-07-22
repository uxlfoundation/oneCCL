OFI_HOME ?= /usr/local/ofi
CFLAGS    = -I$(OFI_HOME)/include -g
LDFLAGS   = -L$(OFI_HOME)/lib -Xlinker -R$(OFI_HOME)/lib -lfabric

TARGETS=sycl_ofi_test

all: $(TARGETS)

sycl_ofi_test: sycl_ofi_test.cpp Makefile
	dpcpp sycl_ofi_test.cpp -o sycl_ofi_test $(CFLAGS) $(LDFLAGS)

clean:
	rm $(TARGETS)
