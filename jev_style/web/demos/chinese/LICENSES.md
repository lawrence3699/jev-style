# Chinese & multilingual demo: sources and licences

| Data | Where it comes from | Licence |
|---|---|---|
| Chinese tickets and e-mails (`cases.json`) | `dev` / `cal` splits of the project pool, `data/master-v3/raw/theme_triage/` (rows with `lang: zh`). The texts are synthetic: they were written for this project by teacher models from structured seeds (row field `license`: `generated:gpt-5.6-sol` or `generated:claude-opus-5.5`). No third-party text. | Project data |
| 51-language utterances (`/demo-data/chinese/massive_dev_subset.json`) | Amazon MASSIVE 1.1, official archive `amazon-massive-dataset-1.1.tar.gz`, **dev partition only**, 52 locales (51 languages), restricted to utterance ids the pool already uses for its `dev`/`cal` rows. | CC BY 4.0 — Copyright Amazon.com Inc. or its affiliates. MASSIVE includes text from SLURP (CC BY 4.0). FitzGerald et al., 2022, "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages". |

Changes: MASSIVE utterances are used unchanged; for each utterance the demo offers the gold intent plus 11
other MASSIVE intent labels as options. The inbox "expected folder" is mapped from the pool's queue label
(`queue_to_folder` in `cases.json`); the urgency, mood, refund and reply questions have no gold answer.

No rows from the training split, the MASSIVE test partition or any evaluation suite are shown.
Hashes and sizes: `data/demos/chinese/manifest.json`. Rebuild: `PYTHONPATH=src:. .venv/bin/python scripts/demos/build_chinese.py`.
