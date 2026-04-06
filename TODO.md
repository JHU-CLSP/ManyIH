# TODO

## Before public release

- [ ] Add LICENSE file (MIT or appropriate license)
- [ ] Update citation BibTeX in README.md with full paper details
- [ ] Clean up evaluator output format — currently dumps verbose internal stats; should output a cleaner summary JSON
- [ ] Add `manyih` console script entry point test (installed via `pip install`, not just `python -m`)
- [ ] Test all three backends end-to-end (OpenRouter, Bedrock, vLLM) with a real model
- [ ] Consider git-lfs for the 20 MB instruction_following.json if hosting on GitHub

## Dataset & Website

- [ ] Release dataset on Hugging Face (upload splits, write dataset card with loading example)
- [ ] Build benchmark website (leaderboard, task descriptions, submission instructions)

## Nice to have

- [ ] Add a few smoke tests (test imports, test data loading, test parse_model_string)
- [ ] Add `--output-dir` to coding evaluator for consistency with IF evaluator
- [ ] Unify CLI arg style (coding uses `--max_samples` with underscore, IF uses `--max-samples` with hyphen)
