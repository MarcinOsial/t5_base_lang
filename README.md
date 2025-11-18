# Resolving Interference When Merging Models ([NeurIPS 2023](https://neurips.cc/))

* Authors: [Prateek Yadav](https://prateeky2806.github.io/), [Derek Tam](https://dptam.github.io/), [Leshem Choshen](https://ktilana.wixsite.com/leshem-choshen), [Colin Raffel](https://colinraffel.com/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* Paper: [arXiv](https://arxiv.org/abs/2306.01708)

<img src="./assets/teis-merging.png" alt="teaser image" width="800"/>


## Setup

1. Create a virtual environment and activate it.
```
python3 -m venv env
source env/bin/activate
```
2. Install dependencies
```
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

3. Download Story Cloze Dataset and update its path in data/dataset_readers.py StoryClozeReader class.

4. Set the path to where finetuned models are stored in utils/merge_utils.py

We have released the IA3 checkpoints [here!](https://drive.google.com/drive/folders/1V2-SLOgK248TQBMP2i_cEdQnxB2jM2E1?usp=sharing)

## Train

### Train T5 Models

```
python src/training.py -c configs/t5_base.json -k train_batch_size=8 gradient_accumulation_factor=1 project_name=training experiment_name=test train_dataset=rte train_dataset_mixture=None num_batches=2
```

## Evaluation

### Evaluate IA3 across multiple prompts and report median.

```
$path_to_checkpoint = # path to your checkpoint
$eval_split = validation
$dataset = rte

python ./src/inference.py -c configs/ia3_base.json --multiple_prompts -i ${dataset} --kwargs checkpoint_to_directly_load_model=${path_to_checkpoint} split=${eval_split} project_name=ia3 experiment_name=${dataset}
```

### Evaluate T5-Large.

```
$path_to_checkpoint = # path to your checkpoint
$eval_split = validation
$dataset = rte

python ./src/inference.py -c configs/t5_large.json -i ${dataset} --kwargs checkpoint_to_directly_load_model=${path_to_checkpoint} split=${eval_split} project_name=t5-large experiment_name=${dataset}
```


## AXIS Training for T5 Models

### Overview

AXIS (Adaptive eXperimental Integration of Singular values) is a method for merging task vectors using Singular Value Decomposition (SVD) and learning optimal singular values. This implementation is specifically designed for T5-base models.

### How It Works

The AXIS training process consists of the following steps:

1. **Task Vector Loading**: Load task vectors for source datasets (fine-tuned models minus base model)
2. **iso_c Merging**: Apply SVD-based merging to combine task vectors:
   - For 2D layers: Extract SVD components (U, S, Vh) and select top components based on `svd_threshold`
   - For non-2D layers: Average task vectors directly
3. **Model Creation**: Create a merged model with learnable singular values:
   - Base T5 model parameters are frozen
   - Only selected singular values (top components) are learnable
   - Remaining components are frozen
4. **Training**: Train learnable singular values on target dataset:
   - Uses AdamW optimizer with learning rate 8e-5
   - Mixed precision training (bfloat16)
   - Validation monitoring every 100 batches (configurable)
   - Early stopping can be enabled (currently disabled by default)
5. **Evaluation**: Evaluate merged model on target dataset test set

### Main Components

- **`axis/t5_task_vectors.py`**: `T5TaskVector` class for loading and normalizing T5 checkpoints
- **`axis/t5_axis_merging.py`**: Main training module with:
  - `iso_c_t5()`: SVD-based merging function
  - `LearnableSingularValuesMergedT5Wrapper`: Model wrapper with learnable singular values
  - `train_t5_axis()`: Training loop
  - `main()`: Entry point with dataset iteration logic

### Configuration

Training configuration is in `axis/configs/t5_axis_training.json`:

```json
{
    "pretrained_model": "t5-base",
    "train_batch_size": 600,
    "eval_batch_size": 600,
    "num_batches": 2000,
    "lr": 8e-5,
    "checkpoint_frequency": 100,
    "early_stopping": false,
    "should_eval_validation": true,
    ...
}
```

### Running Training

#### Using SLURM Script

```bash
sbatch axis/run_t5_axis.sh
```

The script (`axis/run_t5_axis.sh`) handles:
- Environment setup
- Iteration over source dataset combinations
- Multiple SVD thresholds and seeds

#### Direct Python Execution

```bash
python -m axis.t5_axis_merging \
    --svd-threshold=0.1 \
    --model=t5-base \
    --resume-from-idx=0 \
    --end-index=1 \
    --seed=42 \
    --config=axis/configs/t5_axis_training.json
```

**Arguments:**
- `--svd-threshold`: Threshold for SVD component selection (0.0-1.0, default: 0.1)
- `--model`: Base model name (default: t5-base)
- `--resume-from-idx`: Start from this number of source datasets (0 = 1 source)
- `--end-index`: End at this number of source datasets (exclusive)
- `--seed`: Random seed for reproducibility
- `--config`: Path to training configuration JSON
- `--reverse`: (Optional) Reverse dataset order

**Example:**
- `resume-from-idx=0, end-index=1`: Train with 1 source dataset (paws), test on all other datasets
- `resume-from-idx=0, end-index=2`: Train with 1 source (paws), then 2 sources (paws, qasc), test on remaining

### Dataset Order

The datasets are processed in this order:
```
["paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande", "wsc"]
```

### Training Process

1. **Source Dataset Selection**: For each number of source datasets (from `resume-from-idx` to `end-index`):
   - Select source datasets sequentially (e.g., [paws], then [paws, qasc], etc.)
   
2. **Target Dataset Iteration**: For each source combination:
   - Train on all remaining datasets as targets
   - Each target gets a separate training run

3. **Training Steps**:
   - Load base model (t5-base)
   - Load task vectors for source datasets
   - Compute iso_c merging with SVD
   - Create model with learnable singular values
   - Train for `num_batches` batches (default: 2000)
   - Evaluate on target test set
   - Save results to CSV

### Output Files

Results are saved in `exp_out/t5_axis/with_early_stopping/`:

- **CSV file**: `t5_axis_results_{timestamp}_job{slurm_job_id}.csv`
  - Contains: source_datasets, target_dataset, svd_threshold, seed, accuracy, batches_seen, early_stopping_triggered, reverse
  
- **Experiment directory**: `exp_out/t5_axis/t5-axis-{target}-{N}sources-svd{threshold}-seed{seed}/`
  - `results.json`: Full evaluation results
  - `singular_values.json`: Learned singular values
  - `training_log.txt`: Training log (validation accuracy tracking)

### Training Configuration Details

- **Batch Size**: 600 (train and eval)
- **Learning Rate**: 8e-5
- **Optimizer**: AdamW
- **Precision**: bfloat16 (mixed precision)
- **Validation**: Monitored every 100 batches (even if early stopping is disabled)
- **Early Stopping**: Currently disabled by default (can be enabled in config)

### Notes

- Training uses single GPU (no DDP)
- Task vectors are loaded from `exp_out/t5_finetuning/t5-base/`
- Validation uses first 32 samples from validation split
- Test evaluation uses full test split (after removing first 32 samples for validation)

## Merging Models

### T5-Large

#### Basic Averaging
```
$eval_split = validation

python ./src/ties_merging.py -c configs/t5_large.json -i t5_mixture -m t5_mixture -f basic_mean --kwargs split=${eval_split} project_name=t5-large experiment_name=mean
```

#### Task Vectors
```
$eval_split = validation
$eval_function = task-vector_linear+0.1+1.01+0.1

python ./src/ties_merging.py -c configs/t5_large.json -i t5_mixture -m t5_mixture -f ${eval_function} --kwargs split=${eval_split} project_name=t5-large experiment_name=task_vectors
```
Performs merging for different values of lambda. will try out all lambda values between 0 and 1 in incrementso of 0.1.

#### TIES MERGING
```
$eval_split = validation
$redundant = topk20
$elect = mass
$agg = dis-mean
$scale = linear+0.8+2.51+0.1

python ./src/ties_merging.py -c configs/t5_large.json -i t5_mixture -m t5_mixture -f ${redundant}_${elect}_${agg}_${scale} --kwargs split=${eval_split} project_name=t5-large experiment_name=ties
```


### IA3

#### Basic Averaging
```
$eval_split = validation

python ./src/ties_merging.py -c configs/ia3_base.json -i T0_held_out -m T0_held_out -f basic_mean --multiple_prompts --kwargs pretrained_model=bigscience/T0_3B split=${eval_split} project_name=ia3 experiment_name=mean
```

#### Task Vectors
```
$eval_split = validation
$eval_function = task-vector_linear+0.1+1.01+0.1

python ./src/ties_merging.py -c configs/ia3_base.json -i T0_held_out -m T0_held_out -f ${eval_function} --multiple_prompts --kwargs pretrained_model=bigscience/T0_3B split=${eval_split} project_name=ia3 experiment_name=task_vectors
```

#### TIES MERGING
```
$eval_split = validation
$redundant = topk20
$elect = mass
$agg = dis-mean
$scale = linear+0.8+2.51+0.1

python ./src/ties_merging.py -c configs/ia3_base.json -i T0_held_out -m T0_held_out -f ${redundant}_${elect}_${agg}_${scale} --multiple_prompts --kwargs pretrained_model=bigscience/T0_3B split=${eval_split} project_name=ia3 experiment_name=ties
```

# Reference
Please cite our paper if you use our models in your works:


```bibtex
@inproceedings{
      yadav2023tiesmerging,
      title={{TIES}-Merging: Resolving Interference When Merging Models},
      author={Prateek Yadav and Derek Tam and Leshem Choshen and Colin Raffel and Mohit Bansal},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=xtaX3WyCj1}
}
