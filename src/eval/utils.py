import copy
import torch
import os


def prepare_batchOfEvalInfo(batch):
    batchOf_evalInfo = copy.deepcopy(batch)

    for (key, value) in batch.items():
        # Remove ids and mask since no longer needed
        if ("ids" in key) or ("mask" in key):
            del batchOf_evalInfo[key]
        else:
            # Convert tensors to list
            if torch.is_tensor(batchOf_evalInfo[key]):
                batchOf_evalInfo[key] = value.cpu().numpy().tolist()

    return batchOf_evalInfo


def getAndMake_specificPredictionDir(prediction_dir, split, dataset, template_idx):
    """

    Args:
        prediction_dir:
        dataset:
        template_idx:

    Returns:

    """
    # DEBUG: Print all received arguments with their types
    print(f"[DEBUG getAndMake_specificPredictionDir] Received arguments:")
    print(f"  prediction_dir = {prediction_dir} (type: {type(prediction_dir)}, is None: {prediction_dir is None})")
    print(f"  split = {split} (type: {type(split)}, is None: {split is None})")
    print(f"  dataset = {dataset} (type: {type(dataset)}, is None: {dataset is None})")
    print(f"  template_idx = {template_idx} (type: {type(template_idx)}, is None: {template_idx is None})")
    
    # Validate that prediction_dir is not None before using it
    if prediction_dir is None:
        print(f"[ERROR] prediction_dir is None!")
        raise ValueError(
            f"prediction_dir cannot be None in getAndMake_specificPredictionDir. "
            f"This usually means that evaluation_config.prediction_dir was not set properly. "
            f"Check that experiment_dir is set in TrainingConfig and that prediction_dir is "
            f"properly passed to EvaluationConfig."
        )
    
    # Validate that split is not None before using it in os.path.join
    if split is None:
        print(f"[ERROR] split is None!")
        raise ValueError(
            f"split cannot be None in getAndMake_specificPredictionDir. "
            f"This usually means that evaluation_config.split was not set properly. "
            f"Check that split is set in EvaluationConfig (default should be 'validation'). "
            f"Received: split={split}, dataset={dataset}, template_idx={template_idx}"
        )
    
    # Validate that dataset is not None before using it in os.path.join
    if dataset is None:
        print(f"[ERROR] dataset is None!")
        raise ValueError(
            f"dataset (inference_dataset) cannot be None in getAndMake_specificPredictionDir. "
            f"This usually means that evaluation_config.inference_dataset was not set properly. "
            f"Received: split={split}, dataset={dataset}, template_idx={template_idx}"
        )
    
    # Validate that template_idx is not None (it can be 0, but not None)
    if template_idx is None:
        print(f"[ERROR] template_idx is None!")
        raise ValueError(
            f"template_idx cannot be None in getAndMake_specificPredictionDir. "
            f"This usually means that evaluation_config.eval_template_idx was not set properly. "
            f"Received: split={split}, dataset={dataset}, template_idx={template_idx}"
        )

    # DEBUG: Before creating prediction_name
    print(f"[DEBUG] Creating prediction_name from dataset={dataset}, template_idx={template_idx}")
    prediction_name = f"{dataset}_template_{template_idx}"
    print(f"[DEBUG] prediction_name = {prediction_name} (type: {type(prediction_name)})")

    # DEBUG: Before os.path.join - print all values that will be used
    print(f"[DEBUG] Before os.path.join:")
    print(f"  prediction_dir = {prediction_dir} (type: {type(prediction_dir)}, is None: {prediction_dir is None})")
    print(f"  split = {split} (type: {type(split)}, is None: {split is None})")
    print(f"  prediction_name = {prediction_name} (type: {type(prediction_name)}, is None: {prediction_name is None})")
    print(f"[DEBUG] Calling os.path.join(prediction_dir={prediction_dir!r}, split={split!r}, prediction_name={prediction_name!r})")
    
    specificPrediction_dir = os.path.join(prediction_dir, split, prediction_name)
    print(f"[DEBUG] os.path.join result: {specificPrediction_dir}")
    if not os.path.exists(specificPrediction_dir):
        os.makedirs(specificPrediction_dir)

    return specificPrediction_dir


def get_predictionFP(specificPrediction_dir, idx):
    return os.path.join(specificPrediction_dir, f"run_{idx}.txt")


def get_dirAndRunIdx_fromPredictionFp(prediction_fp):
    """

    Args:
        prediction_fp:

    Returns:

    """
    directory = os.path.dirname(prediction_fp)
    run_filename = os.path.basename(prediction_fp)
    run_idx = int(run_filename.replace("run_", "").replace(".txt", ""))
    return directory, run_idx
