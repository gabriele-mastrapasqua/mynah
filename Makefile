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

SRC := $(wildcard src/*.c) vendor/cJSON.c
OBJ := $(SRC:.c=.o)
HDR := $(wildcard src/*.h)

MODEL_DIR ?= models/nemotron-3.5-asr-streaming-0.6b

all: mynah

mynah: $(OBJ) cli/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

tests/test_features: tests/test_features.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Parità C vs oracolo. Skip (exit 77) se mancano modello o dump golden.
# Rigenera i dump con: make golden-dump
test: tests/test_features
	@tests/test_features $(MODEL_DIR) tests/audio/test_it.wav tests/golden/test_it || \
	  ( [ $$? -eq 77 ] && echo "SKIP: modello o golden assenti (make golden-dump)" )

golden-dump:
	cd tools && uv run python -m oracle.transcribe ../$(MODEL_DIR) ../tests/audio/test_it.wav \
	  --lang it-IT --dump-dir ../tests/golden/test_it

clean:
	rm -f mynah $(OBJ) cli/main.o tests/test_features tests/test_features.o

.PHONY: all clean test golden-dump
