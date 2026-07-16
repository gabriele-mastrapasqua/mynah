# Mynah — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -Isrc
LDFLAGS ?=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate
  CFLAGS  += -DMYNAH_BLAS_ACCELERATE -DACCELERATE_NEW_LAPACK
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

TESTS := tests/test_features tests/test_subsampling tests/test_encoder

tests/%: tests/%.o tests/npy.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Parità C vs oracolo. Skip (exit 77) se mancano modello o dump golden.
# Rigenera i dump con: make golden-dump
test: $(TESTS) mynah
	@for t in $(TESTS); do \
	  $$t $(MODEL_DIR) tests/audio/test_it.wav tests/golden/test_it; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP $$t: modello o golden assenti (make golden-dump)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@sh tests/test_e2e.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP e2e: modello assente"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

golden-dump:
	cd tools && uv run python -m oracle.transcribe ../$(MODEL_DIR) ../tests/audio/test_it.wav \
	  --lang it-IT --dump-dir ../tests/golden/test_it

clean:
	rm -f mynah $(OBJ) cli/main.o tests/*.o $(TESTS)

.PHONY: all clean test golden-dump
