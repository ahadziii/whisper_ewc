# Whisper EWC

This project integrates Elastic Weight Consolidation (EWC), a Lifelong Learning technique, in the Whisper model to mitigate catastrophic forgetting.

## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ahadziii/whisper_ewc.git
    cd whisper_ewc
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## How to Run the Script

To run the script, use the following command:
```sh
python torchrun --nproc_per_node=1 finetune_whisper.py \
    --prev_models <PREV_MODELS_PATH> \
    --fisher_audio_paths_files <FISHER_AUDIO_PATHS_FILES> \
    --fisher_transcriptions_files <FISHER_TRANSCRIPTIONS_FILES> \
    --lambda_ewc <LAMBDA_EWC> \
    --fisher_samples <FISHER_SAMPLES> \
    --device <DEVICE> \
    --model_name <MODEL_NAME> \
    --sampling_rate <SAMPLING_RATE> \
    --num_proc <NUM_PROC> \
    --train_strategy <TRAIN_STRATEGY> \
    --learning_rate <LEARNING_RATE> \
    --warmup <WARMUP> \
    --train_batchsize <TRAIN_BATCHSIZE> \
    --eval_batchsize <EVAL_BATCHSIZE> \
    --num_epochs <NUM_EPOCHS> \
    --resume_from_ckpt <RESUME_FROM_CKPT> \
    --output_dir <OUTPUT_DIR> \
    --train_datasets <TRAIN_DATASETS> \
    --eval_datasets <EVAL_DATASETS>
```

