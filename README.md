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
python main.pytorchrun --nproc_per_node=<NUM_GPUS> <SCRIPT_NAME> \
    --model_name <MODEL_NAME> \
    --sampling_rate <SAMPLING_RATE> \
    --num_proc <NUM_PROC> \
    --train_strategy <TRAIN_STRATEGY> \
    --learning_rate <LEARNING_RATE> \
    --warmup <WARMUP_STEPS> \
    --train_batchsize <TRAIN_BATCHSIZE> \
    --eval_batchsize <EVAL_BATCHSIZE> \
    --num_epochs <NUM_EPOCHS> \
    --resume_from_ckpt <CHECKPOINT_PATH> \
    --output_dir <OUTPUT_DIRECTORY> \
    --train_datasets <TRAIN_DATASET_PATH> \
    --eval_datasets <EVAL_DATASET_PATH>
```

