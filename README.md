# Code for paper "Analog foundation models"

## Setup

```bash
conda create -n analog-foundation-models python=3.10 -y
conda activate analog-foundation-models
# GCC needs to be at least >=9 for xformers compilation.
# module load gcc/9.3.0/1  # necessary on some clusters
pip install xformers
pip install vllm
pip install tqdm datasets
```

## Quickstart
The file [example.py](example.py) shows how to use the pre-trained analog foundation models.
To let the model generate in response to a prompt, simply do
```bash
python example.py --model-name meta-llama/Llama-3.2-1B-Instruct
```
or 
```bash
python example.py --model-name microsoft/Phi-3-mini-4k-instruct
```

One can also customize the prompt with `--prompt "some prompt"`.

## Creating an analog foundation model

Make sure that your pwd is this repository. We will generate data and train
on the data in a distributed fashion using DeepSpeed. This tutorial is
for `"meta-llama/Llama-3.2-1B-Instruct"`, but works exactly the same for the
`"microsoft/Phi-3-mini-4k-instruct"` model.

### Model download
It is generally better to download the models to disk and then load them from
disk when doing training/generation across 100s of GPU processes.
```bash
python
>> import os; from transformers import AutoModelForCausalLM, AutoTokenizer
>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
>> model.save_pretrained("./data/meta-llama/Llama-3.2-1B-Instruct")
>> tokenizer.save_pretrained("./data/meta-llama/Llama-3.2-1B-Instruct")
```
The model is now saved to the `data/meta-llama/Llama-3.2-1B-Instruct` folder in this repository.

### Data generation
First, run the data generation script on one GPU alone to check that everything is working.
```bash
python curate_synthetic_dataset.py \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --model-base-path "./data" \
    --save-base-path "./data" \
    --num-gpus-per-node 1 \
    --num-nodes 1 \
    --node-id 0 \
    --number-of-tokens 20480
```

Then, post-process the data using
```bash
python post_process_synthetic_dataset.py \
    --workers-path ./data/synthetic-dataset-meta-llama/Llama-3.2-1B-Instruct \
    --save-path ./data/processed-data/meta-llama/Llama-3.2-1B-Instruct
```

### Inspecting the dataset
```bash
python
>> from transformers import AutoTokenizer
>> from datasets import load_from_disk
>> tokenizer = AutoTokenizer.from_pretrained("./data/meta-llama/Llama-3.2-1B-Instruct")
>> ds = load_from_disk("./data/synthetic-dataset-meta-llama/Llama-3.2-1B-Instruct/worker-0")
>> print(tokenizer.decode(ds[0]["input_ids"]))
```

### Training
Install some additional packages:
```bash
pip install accelerate wandb
```

Install [AIHWKIT-Lightning](https://github.com/IBM/aihwkit-lightning):
```bash
git clone https://github.com/IBM/aihwkit-lightning.git
cd aihwkit-lightning
pip install -r requirements.txt
pip install -e .
```
Also, follow the steps [here](https://github.com/IBM/aihwkit-lightning/tree/main/examples/deepspeed_and_huggingface) to adapt the DeepSpeed ZeRO optimizer to the input range learning. It is recommended to clone and locally install DeepSpeed.

To quickly try out the training, only one GPU is required (even one V100 suffices).

```bash
accelerate launch \
    --use_deepspeed \
    --deepspeed_hostfile ./data/training_files/hostfile \
    --deepspeed_multinode_launcher standard \
    --dynamo_backend no \
    --mixed_precision fp16 \
    --num_processes 1 \
    --gpu_ids all \
    --num_machines 1 \
    --machine_rank 0 \
    --rdzv_backend static \
    --deepspeed_config_file ./data/training_files/ds_config.json \
    ./src/analog_foundation_models/train.py --config ./data/training_files/config.yaml
```

### Re-training the reported analog foundation models
To speed up the process of data generation, one should use a cluster with multiple
nodes, each comprising for example 8 GPUs. Every cluster setup is different, but we
present a simple SLURM script `submit_synthetic_generation_jobs.sh`:


```bash
#!/bin/bash
#SBATCH --output=synth-generation-log/%j.out
#SBATCH --error=synth-generation-log/%j.err
#SBATCH --time=120
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=npl-2024

# you might not need this!
module load gcc/9.3.0/1

conda activate analog-foundation-models

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${{SLURM_JOB_ID}}"
python curate_synthetic_dataset.py \
    --model-name $1 \
    --model-base-path "./data" \
    --save-base-path "./data" \
    --num-gpus-per-node 8 \
    --num-nodes $2 \
    --node-id $3 \
    --number-of-tokens $4
echo "Finished at:      $(date)"
```

which can be started from a script like this:
```bash
#!/bin/bash
# Set the number of nodes and number of tokens
NUM_NODES=1200
NUM_TOKENS=20000000000
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct

# Loop from 0 to NUM_NODES - 1
for (( i=0; i<NUM_NODES; i++ ))
do
    # Submit the job with sbatch
    sbatch submit_synthetic_generation_jobs.sh $MODEL_NAME $NUM_NODES $i $NUM_TOKENS
done
```

This will create a bunch of "worker" folders in the `./data/synthetic-dataset-{MODEL_NAME}` folder,
which can then be post-processed by `post_process_synthetic_dataset.py` (see above).

To train on more nodes and GPUs, you can find an example setup [here](data/training_files/meta-llama/Llama-3.2-1B-Instruct/),
where the [script.sh](data/training_files/meta-llama/Llama-3.2-1B-Instruct/script.sh) can be used to start
distributed training using DeepSpeed. If you have `NNODES` and `NGPUS_PER_NODE`, you need to adapt these lines in the script
accordingly:\
`#SBATCH --nodes=1` → `#SBATCH --nodes=NNODES`\
`#SBATCH --gres=gpu:8` → `#SBATCH --gres=gpu:NGPUS_PER_NODE`\
`--num_machines 1\` → `--num_machines NNODES\`\
`--num_processes 8\` → `--num_processes NNODES*NGPUS_PER_NODE\`\
Finally, you can submit the job using `sbatch script.sh`.
