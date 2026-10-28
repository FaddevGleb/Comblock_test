# Latest hybrid training/inference package

The complete reproducible package is stored as the Git LFS artifact:

`artifacts/hybrid_solution_latest_package_20261028.zip`

Archive SHA-256:

`bb3878e7bd6f137dbc3791edcd45931b5e057a395633e27fe34f07a249fb5f7b`

The archive contains the latest SFT/QLoRA and DPO code, synthetic/preference data, skill and stylometry extraction (`auto-style-capture`), LUAR training code/data/model, SFT adapters, inference runner and the completed `run_v4_temp05` output.

The checked-in files under this directory expose the most important lightweight artifacts directly: the current inference runner, compact/full style masks, LUAR training inputs, run manifests and the latest generation report/results.

Current inference settings: `temperature=0.5`, `top_p=0.95`, eight candidates per news item. `banki_oil` uses the compact type1 news-corpus mask; `moscowach` uses its adapter-training system prompt.
