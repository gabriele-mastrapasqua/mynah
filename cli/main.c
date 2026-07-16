/* mynah — CLI. Per ora solo stub: version e usage.
 * I sottocomandi arrivano con le milestone: transcribe (M1.2), stream (M1.3). */
#include <stdio.h>
#include <string.h>
#include "mynah.h"

static void usage(void) {
    printf("mynah %s — native ASR runtime for NeMo speech models\n\n", mynah_version());
    printf("Uso: mynah <comando> [opzioni]\n\n");
    printf("Comandi (in sviluppo, vedi TODO.md):\n");
    printf("  transcribe -m <model_dir> -i <file.wav>   trascrizione offline   [M1.2]\n");
    printf("  stream     -m <model_dir>                 streaming da stdin     [M1.3]\n");
    printf("  --version                                 stampa la versione\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--version") == 0) {
        printf("%s\n", mynah_version());
        return 0;
    }
    usage();
    return argc < 2 ? 0 : 1;
}
