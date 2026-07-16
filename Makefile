# Mynah — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -Isrc
LDFLAGS ?=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate
  CFLAGS  += -DMYNAH_BLAS_ACCELERATE
else
  LDFLAGS += -lopenblas -lm -lpthread
  CFLAGS  += -DMYNAH_BLAS_OPENBLAS
endif

SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)

all: mynah

mynah: $(OBJ) cli/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c src/mynah.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f mynah $(OBJ) cli/main.o

# test: unit test C (arrivano con M0.5+)
# test-golden: WER vs trascrizioni NeMo (arriva con M1.2)

.PHONY: all clean
