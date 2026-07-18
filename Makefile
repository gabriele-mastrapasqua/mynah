# Mynah — build. CPU-first: BLAS = Accelerate (macOS) / OpenBLAS (Linux).
CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -Isrc
LDFLAGS ?=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation
  BLAS_DEF := MYNAH_BLAS_ACCELERATE
  CFLAGS  += -DMYNAH_BLAS_ACCELERATE -DACCELERATE_NEW_LAPACK -DMYNAH_METAL
  OBJ_EXTRA := src/metal_mps.o
else
  LDFLAGS += -lopenblas -lm -lpthread
  BLAS_DEF := MYNAH_BLAS_OPENBLAS
  CFLAGS  += -DMYNAH_BLAS_OPENBLAS
endif

SRC := $(wildcard src/*.c) vendor/cJSON.c
OBJ := $(SRC:.c=.o) $(OBJ_EXTRA)
HDR := $(wildcard src/*.h)

MODEL_DIR ?= models/nemotron-3.5-asr-streaming-0.6b
PARAKEET_DIR ?= models/parakeet-tdt-0.6b-v3
PARAKEET110_DIR ?= models/parakeet-tdt_ctc-110m

all: mynah mynah-server

mynah: $(OBJ) cli/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mynah-server: $(OBJ) server/main.o server/http_util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

src/metal_mps.o: src/metal_mps.m
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@

TESTS := tests/test_qmat tests/test_features tests/test_subsampling tests/test_encoder tests/test_streaming tests/test_batch

tests/%: tests/%.o tests/npy.o tests/testcfg.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Parità C vs oracolo (Nemotron streaming + Parakeet TDT offline).
# Skip (exit 77) se mancano modello o dump golden. Rigenera con: make golden-dump
PARITY_BOTH := tests/test_features tests/test_subsampling tests/test_encoder tests/test_batch
test: $(TESTS) mynah examples/minimal
	@for t in $(TESTS); do \
	  if [ $$t = tests/test_qmat ]; then $$t; rc=$$?; \
	  else $$t $(MODEL_DIR) tests/audio/test_it.wav tests/golden/test_it; rc=$$?; fi; \
	  if [ $$rc -eq 77 ]; then echo "SKIP $$t: modello o golden assenti (make golden-dump)"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done
	@for spec in "$(PARAKEET_DIR) tests/audio/test_it.wav tests/golden/parakeet_it" \
	             "$(PARAKEET110_DIR) tests/audio/test_en.wav tests/golden/parakeet110_en"; do \
	  for t in $(PARITY_BOTH); do \
	    $$t $$spec; rc=$$?; \
	    if [ $$rc -eq 77 ]; then echo "SKIP $$t ($$spec): modello o golden assenti"; \
	    elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	  done; \
	done
	@for m in $(MODEL_DIR) $(PARAKEET_DIR) $(PARAKEET110_DIR) \
	          models/parakeet-rnnt-0.6b models/parakeet-ctc-0.6b \
	          models/canary-180m-flash; do \
	  sh tests/test_e2e.sh $$m; rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP e2e: $$m assente"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi; \
	done

golden-dump:
	cd tools && uv run python -m oracle.transcribe ../$(MODEL_DIR) ../tests/audio/test_it.wav \
	  --lang it-IT --dump-dir ../tests/golden/test_it
	@if [ -f $(PARAKEET_DIR)/mynah.json ]; then \
	  cd tools && uv run python -m oracle.transcribe ../$(PARAKEET_DIR) \
	    ../tests/audio/test_it.wav --dump-dir ../tests/golden/parakeet_it; fi
	@if [ -f $(PARAKEET110_DIR)/mynah.json ]; then \
	  cd tools && uv run python -m oracle.transcribe ../$(PARAKEET110_DIR) \
	    ../tests/audio/test_en.wav --dump-dir ../tests/golden/parakeet110_en; fi

# libreria statica (senza CLI)
lib: libmynah.a
libmynah.a: $(OBJ)
	ar rcs $@ $^

# esempio API (compilato in `make test`: guardia sulla superficie pubblica)
example: examples/minimal
examples/minimal: examples/minimal.c libmynah.a
	$(CC) $(CFLAGS) -o $@ examples/minimal.c libmynah.a $(LDFLAGS)

# CUDA (Linux, richiede nvcc): GEMM grandi su GPU. NON validato su hardware:
# compilare e lanciare `make test` sulla macchina CUDA prima di fidarsi.
NVCC ?= nvcc
cuda:
	$(MAKE) clean && $(MAKE) CFLAGS="$(CFLAGS) -DMYNAH_CUDA" \
	  OBJ_EXTRA="src/cuda_gemm.o" \
	  LDFLAGS="$(LDFLAGS) -lcublas -lcudart -L/usr/local/cuda/lib64"

src/cuda_gemm.o: src/cuda_gemm.cu
	$(NVCC) -O3 -DMYNAH_CUDA -c $< -o $@

# build alternative.
# Policy memoria/UB su macOS: `make leaks` (nativo, veloce) + `make ubsan` (overhead
# basso). ASan è LENTISSIMO su Mac e tende a impallarsi col modello grande: solo CI Linux.
debug:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O0 -g -Wall -Wextra -Isrc -D$(BLAS_DEF)"
ubsan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O2 -g -fsanitize=undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -Isrc -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=undefined" all test
asan:
	$(MAKE) clean && $(MAKE) CFLAGS="-std=c11 -O1 -g -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -Wall -Wextra -Isrc -D$(BLAS_DEF) -DACCELERATE_NEW_LAPACK" \
	  LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined" all test

# bench riproducibile: RTF warm + picco RAM per ogni modello presente
bench: mynah
	@sh tests/bench.sh

# Test end-to-end del server (REST + concorrenza + WebSocket)
test-server: mynah-server
	@sh tests/test_server.sh $(MODEL_DIR); rc=$$?; \
	  if [ $$rc -eq 77 ]; then echo "SKIP test-server: modello assente"; \
	  elif [ $$rc -ne 0 ]; then exit $$rc; fi

# Suite multilingua: sample audio reali (Tatoeba, CC) per ogni lingua supportata,
# verifica language detection + CER vs testo di riferimento.
# Prima volta: make fetch-lang-samples (richiede ffmpeg + tools/ uv).
test-nemo-langs: mynah
	cd tools && uv run python -m eval.test_langs

fetch-lang-samples:
	cd tools && uv run python fetch_lang_samples.py 3

# leak check veloce su macOS (tool nativo `leaks`, nessuna rebuild — su Mac ASan
# è lentissimo: usarlo solo in CI Linux. Stesso pattern di qwen-tts).
leaks: mynah tests/test_streaming
	leaks --atExit -- ./mynah transcribe -m $(MODEL_DIR) -i tests/audio/test_it.wav \
	  --lang it-IT 2>&1 | tail -3
	leaks --atExit -- tests/test_streaming $(MODEL_DIR) tests/audio/test_it.wav \
	  tests/golden/test_it 2>&1 | tail -3

clean:
	rm -f mynah mynah-server libmynah.a $(OBJ) cli/main.o server/*.o tests/*.o $(TESTS) \
	  examples/minimal

.PHONY: all clean test golden-dump lib debug ubsan asan bench leaks test-nemo-langs fetch-lang-samples test-server cuda
