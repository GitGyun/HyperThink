NUM_SHARDS=8 bash scripts/eval/distributed_evaluate.sh --ckpt outputs/checkpoints/smollm3-3b-extended.ckpt --sm3b --task aime_24_25
NUM_SHARDS=8 bash scripts/eval/distributed_evaluate.sh --ckpt outputs/checkpoints/smollm3-3b-extended.ckpt --sm3b --task commonsenseqa
NUM_SHARDS=8 bash scripts/eval/distributed_evaluate.sh --ckpt outputs/checkpoints/smollm3-3b-extended.ckpt --sm3b --task bbh
NUM_SHARDS=8 bash scripts/eval/distributed_evaluate.sh --ckpt outputs/checkpoints/smollm3-3b-extended.ckpt --sm3b --task livecodebench