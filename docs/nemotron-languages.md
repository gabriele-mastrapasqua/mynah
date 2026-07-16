# Nemotron 3.5 ASR Streaming — lingue supportate

> Fonte: model card HF (tier) + `processor_config.json` reale (`prompt_dictionary`,
> in `reference/nemotron-3.5-asr-streaming-0.6b/`). Aggiornato 2026-07-16.

Il modello supporta **40 locale ufficiali** divisi in 3 tier di qualità, più language
detection automatica (`target_lang=auto`, prompt id 101). Il `prompt_dictionary` contiene
in realtà **105 slot** (id 0–104): gli slot oltre i 40 ufficiali servono per fine-tuning
su nuove lingue (vedi blog NVIDIA sul fine-tuning).

## Tier 1 — Transcription-ready (19 locale)

Qualità pronta per produzione. WER FLEURS @1.12s tra parentesi dove pubblicato.

| Locale | Lingua | Prompt id | Note |
|---|---|---|---|
| en-US | Inglese (USA) | 0 | WER 7.91% |
| en-GB | Inglese (UK) | 1 | |
| es-ES | Spagnolo (Spagna) | 2 | |
| es-US | Spagnolo (USA) | 3 | WER 4.11% |
| fr-FR | Francese | 8 | |
| fr-CA | Francese (Canada) | 100 | |
| **it-IT** | **Italiano** | **15** | **WER 4.25%** |
| pt-BR | Portoghese (Brasile) | 12 | WER 5.48% |
| pt-PT | Portoghese (Portogallo) | 13 | |
| nl-NL | Olandese | 16 | |
| de-DE | Tedesco | 9 | |
| tr-TR | Turco | 18 | |
| ru-RU | Russo | 11 | |
| ar-AR | Arabo | 7 | |
| hi-IN | Hindi | 6 | |
| ja-JP | Giapponese | 10 | |
| ko-KR | Coreano | 14 | |
| vi-VN | Vietnamita | 33 | |
| uk-UA | Ucraino | 19 | |

## Tier 2 — Broad-coverage (13 locale)

Copertura ampia, qualità inferiore (media tier: WER FLEURS 22.13%).

pl-PL (17) · sv-SE (24) · cs-CZ (22) · nb-NO (103) · da-DK (25) · bg-BG (30) · fi-FI (26) ·
hr-HR (29) · zh-CN (4) · hu-HU (23) · ro-RO (20) · sk-SK (28) · et-EE (60)

## Tier 3 — Adaptation-ready (8 locale)

Supportati ma pensati per fine-tuning prima dell'uso in produzione.

el-GR (21) · lt-LT (31) · lv-LV (61) · mt-MT (102) · sl-SI (62) · he-IL (64) · th-TH (32) ·
nn-NO (104)

## Language detection e tag in output

- Con `auto` (id 101) il modello rileva la lingua e **emette il tag locale** (es. `<it-IT>`)
  come token nell'output, dopo la punteggiatura terminale — il runtime lo strippa dal testo
  e lo espone come `mynah_result.lang`.
- I 39 tag lingua sono token speciali del vocabolario (es. `<it-IT>` = id 1279); l'elenco
  completo è negli `added_tokens` di `tokenizer.json`.
- Alias accettati nel dizionario: forme corte (`it`, `de`, `fr`, …) mappano sullo stesso id
  del locale principale.

## Extra (slot presenti nel dizionario ma NON tra i 40 ufficiali)

Il `prompt_dictionary` include anche id per: zh-TW, id-ID, ms-MY, fa-IR, ur-PK, bn-IN,
ta-IN, te-IN, kn-IN, ml-IN, gu-IN, mr-IN, ne-NP, si-LK, km-KH, sw-KE, am-ET, ha-NG, yo-NG,
ig-NG, zu-ZA, af-ZA, e altri (~65 slot totali oltre i 40). Sono slot di training/fine-tuning:
qualità non dichiarata, da non esporre come "supportati" senza test.

## Punteggiatura e maiuscole

Native nell'output per tutte le lingue (niente post-processing).
