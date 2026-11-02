"""Building blocks for the DataMate baseline.

- `tables`:  read one shop's CSV / XLSX / SQLite files as raw grids.
- `sandbox`: run a model-written pandas snippet in a separate process.
- `agent`:   prompt the model, execute its snippet, repair it on failure.
- `llm`:     settings (`Config`) and the shared chat-completion call.
- `scoring`: answer format gate and the offline Penalized Accuracy scorer.
- `jsonl`:   append-only JSONL reading and writing shared by the entrypoints.
"""
