#!/bin/bash
# KGAT cuisine A/B on the unextended (graph.pkl-reconstructed) dataset.
# Arm A: original Google-types cuisine.  Arm B: LLM-inferred cuisine.
# Both: emb 2048, 300 epochs, dual-GPU DDP, no prebuilt features (cuisine flows from
# restaurants_enriched into BOTH the item one-hot and the KGAT knowledge graph).
set -u
cd /home/swami/Work/Projects/foodie_revamp
export FOODIE_REVIEWS_FLAT=data/reviews_flat_unextended.parquet
COMMON="--epochs 300 --emb-dim 2048 --eval-every 50 --no-spatial-cbg --save"

echo "[A baseline / types cuisine] start $(date)"
FOODIE_RESTAURANTS_ENR=data/restaurants_enriched.parquet \
torchrun --nproc_per_node=2 --master_port=29611 recommendation_kgat.py $COMMON \
  --run-tag unext_typescz > exp_logs/kgat_typescz.log 2>&1
echo "[A] done $(date) exit=$?"

echo "[B llm cuisine] start $(date)"
FOODIE_RESTAURANTS_ENR=data/restaurants_enriched_llmcz.parquet \
torchrun --nproc_per_node=2 --master_port=29612 recommendation_kgat.py $COMMON \
  --run-tag unext_llmcz > exp_logs/kgat_llmcz.log 2>&1
echo "[B] done $(date) exit=$?"
echo "[AB COMPLETE] $(date)"
