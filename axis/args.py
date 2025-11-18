"""
Argument list

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import argparse
import os

import torch

def int_or_float(value):
    if '.' in value:
        return float(value)
    return int(value)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default="/raid/NFS_SHARE/home/marcin.osial/merg_datasets",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--eval-on-full",
        default=False,
        action="store_true",
        help="Evaluate on the full dataset, when the model is trained on one class."
    )
    parser.add_argument(
        "--loss-fn",
        default='entropy',
        type=str,
        help="Loss function to use.",
        choices=["entropy", "cross_entropy"]
    )
    parser.add_argument(
        "--ind-dataset",
        default=None,
        type=str,
        help="Which dataset to use for starting the learning. ",
    )
    parser.add_argument(
        "--number-of-random-matrices",
        type=int,
        default=3,
        help="Number of random matrices to generate for each task vector.",
    )
    parser.add_argument(
        "--lp-reg",
        default=None,
        type=int,
        choices=[1, 2],
        help="Regularisation applied to the learned coefficients."
    )
    parser.add_argument(
        "--component-selection-loops",
        type=int,
        default=16,
        help="Number of loops to select components for iso_c."
    )
    parser.add_argument(
        "--blockwise-coef",
        default=False,
        action="store_true",
        help="Use different coefficients on different parameter blocks."
    )
    parser.add_argument(
        "--subsample",
        default=1.0,
        type=int_or_float,
        help="Subsample the datasets with a float or specify the number of shots with an integer."
    )
    parser.add_argument(
        "--start-from-block-number",
        type=int,
        default=0,
        help="Start from this block number.",
    )
    parser.add_argument(
        "--control-threshold",
        default=0.95,
        type=float,
        help="Percentage of accuracy on the control dataset to maintain."
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--source-dataset-name",
        type=str,
        default=None,
        help="The name of the single source dataset to use for creating the task vector.",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=None,
        required=False,
        help="Source dataset used to initialize the model for full fine-tuning.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default="/raid/NFS_SHARE/home/marcin.osial/atlas/models/",
        help="Where to load zero-shot weights and task vectors",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default='/raid/NFS_SHARE/home/marcin.osial/atlas/models/logdir/',
        help="Where to save results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Adapter trained with aTLAS",
        choices=["tip", "lpp", "tip_cot"],
    )
    parser.add_argument(
        "--finetuning-mode",
        default='standard',
        choices=["standard", "linear", "posthoc", "none"],
        help="Whether to use linearized models or not.",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="Percentage of trainable parameters to use for the experiment.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Run atlas x K where the task vectors are randomly partitioned n times (few-shot only)",
    )
    parser.add_argument(
        "--total-trainable-params",
        type=int,
        default=None,
        help="Specify the exact total number of trainable coefficients. Overrides --partition and --blockwise-coef if set.",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        default=False,
        help="Skip automatic git commit at the beginning of the experiment run",
    )
    parser.add_argument(
        "--target-dataset-name",
        type=str,
        default="MNIST",
        help="Target dataset for task arithmetic",
    )
    parser.add_argument(
        "--iso_c_component_retention_ratio",
        type=float,
        default=0.1,
        help="Ratio of SVD components to retain during iso_c merging (e.g., 0.1 for top 10%). Used if projection error ranking is active."
    )
    parser.add_argument(
        "--learnable_sv_ratio",
        type=float,
        default=0.1,
        help="Ratio of singular values to make learnable in LearnableSingularValuesMergedEncoder (e.g., 0.1 for top 10%)."
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=76,
        help="Number of top components to keep in iso_c."
    )
    parser.add_argument(
        "--max-num-components",
        type=int,
        default=76,
        help="Maximum number of components to select with Orthogonal Matching Pursuit (OMP)."
    )
    parser.add_argument(
        "--activation_corruption_name",
        type=str,
        default="defocus_blur",
        choices=[
            "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", 
            "gaussian_blur", "glass_blur", "defocus_blur", "motion_blur", 
            "zoom_blur", "snow", "spatter", "contrast", "brightness", 
            "saturate", "jpeg_compression", "pixelate", "elastic_transform",
            "none"
        ],
        help="Name of the corruption to apply during activation collection for iso_c. Set to 'none' for no corruption."
    )
    parser.add_argument(
        "--activation_corruption_severity",
        type=int,
        default=5,
        choices=range(1, 6),
        metavar='[1-5]',
        help="Severity of the corruption applied during activation collection (1-5)."
    )
    parser.add_argument(
        "--max_activation_batches",
        type=lambda x: x if x.lower() == 'max' else int(x),
        default=16,
        help="Maximum number of batches to process for activation collection for iso_c. Set to 'max' to process all batches from the activation dataset."
    )
    parser.add_argument(
        "--svd-threshold",
        type=float,
        default=0,
        help="Threshold for zeroing out top singular values in SVD.",
    )
    parser.add_argument(
        "--svd-threshold-first",
        type=float,
        default=0,
        help="Threshold for zeroing out top singular values in SVD (first stage).",
    )
    parser.add_argument(
        "--svd-threshold-second",
        type=float,
        default=0,
        help="Threshold for zeroing out top singular values in SVD (second stage).",
    )
    parser.add_argument(
        "--no-use-half",
        action="store_true",
        default=False,
        help="If set, don't use half precision (float16) for task vectors. Use full precision (float32) instead, which is needed for SVD operations.",
    )
    parser.add_argument(
        "--keep-top-values",
        action="store_true",
        default=False,
        help="If set, keep top singular values and zero out the rest. If not set (default), zero out top singular values and keep the rest.",
    )
    parser.add_argument(
        "--sorting-descending",
        action="store_true",
        default=False,
        help="If set, sort singular values in descending order for iso_c selection. Defaults to False (ascending).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=24,
        help="Index to end at.",
    )
    parser.add_argument(
        "--resume-from-idx",
        type=int,
        default=0,
        help="Index to resume from.",
    )
    parser.add_argument('--memory-efficient', action='store_true', 
                   help='Enable memory optimizations for large partition sizes')
    parser.add_argument(
        "--isoc",
        default=False,
        action="store_true",
        help="Use the iso_c function to merge task vectors with SVD-based averaging."
    )
    parser.add_argument(
        "--svd-component-index",
        type=int,
        default=1, # Default to second best component (index 1)
        help="0-based index of the singular value component to select from each task (0=best, 1=second best, etc.)."
    )
    parser.add_argument(
        "--save-learned-params-to",
        type=str,
        default=None,
        help="Path to save learned parameters and metadata."
    )
    parser.add_argument(
        "--load-learned-params-from",
        type=str,
        default=None,
        help="Path to load learned parameters and metadata."
    )
    parser.add_argument(
        "--load-only-classification-head",
        action="store_true",
        default=False,
        help="If set, only the classification head will be loaded from the checkpoint file."
    )
    parser.add_argument(
        "--iso-c-ranking-selection-component-number",
        type=int,
        default=-1,
        help="Component number to select during iso_c ranking selection. Default: -1 (use all or default behavior)."
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default="impulse_noise",
        choices=CORRUPTION_NAMES,
        help="Type of corruption to apply."
    )
    parser.add_argument(
        "--severity-number",
        type=int,
        default=5,
        help="Severity of the corruption applied during evaluation (1-5)."
    )
    parser.add_argument(
        "--max-activation-batches",
        type=int,
        default=16,
        help="Maximum number of batches to process for activation collection for iso_c."
    )
    parser.add_argument(
        "--num_noise_augmented_samples_per_task",
        type=int,
        default=32,
        help="Number of noise-augmented samples per task for iso_c augmentation. K in the paper."
    )
    parser.add_argument(
        "--augmentation_initial_noise_std",
        type=float,
        default=0.1,
        help="Initial standard deviation for noise augmentation."
    )
    parser.add_argument(
        "--augmentation_noise_std_increase_factor",
        type=float,
        default=1.5,
        help="Factor by which to increase noise standard deviation for augmentation."
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args

def get_corruption_names():
    return [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "gaussian_blur",
        "glass_blur",
        "defocus_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "spatter",
        "contrast",
        "brightness",
        "saturate",
        "jpeg_compression",
        "pixelate",
        "elastic_transform",
    ]

CORRUPTION_NAMES = get_corruption_names()
