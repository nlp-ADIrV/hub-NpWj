#!/usr/bin/env bash
set -e
python src/run_baselines.py   --data_dirs data/bq_corpus_sample data/lcqmc_sample   --split validation   --output_dir outputs
