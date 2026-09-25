# Playground presets: data sources and licences

`presets.json` is built by `scripts/demos/build_playground_presets.py`. Every preset is one case
from the **dev or cal** split of the project pool (`data/master-v3/raw/<pool>/{dev,cal}.jsonl`).
No train rows and no sealed evaluation suites are used. Each preset lists its row ids in
`source.row_ids`, and the dataset's own label in `reference`.

| Preset | Pool | Split | Original dataset | Licence |
|---|---|---|---|---|
| agent-trace, invoice, security, support | typed_official | dev | LocalLLaMA/typed-decisions | Apache-2.0 |
| triage-zh | theme_triage | cal | Generated for this project by a teacher model | Project-generated text |
| intent-zh | intent_massive | dev | AmazonScience/MASSIVE 1.1 (zh-CN) | CC BY 4.0 |
| nli | nli | dev | MultiNLI (validation_matched, genre "slate") | OANC-derived genres are free to use, modify and share (MultiNLI data card) |
| relevance | theme_relevance | dev | SQuAD 2.0 (rajpurkar/squad_v2) | CC BY-SA 4.0 |
| injection | theme_jailbreak | dev | yanismiraoui/prompt_injections | Apache-2.0 |

Attribution: MASSIVE (FitzGerald et al., 2022); MultiNLI (Williams et al., 2018); SQuAD 2.0
(Rajpurkar et al., 2018). The SQuAD passage is shown unchanged under CC BY-SA 4.0.

No code, text or assets from any other product are used on these pages.
