from evaluate import load
from datasets import DownloadConfig

from src.utils.utils import convert_dictOfLists_to_listOfDicts, get_average


def _patch_download_config_for_compatibility():
    """
    Apply monkey patches to DownloadConfig to ensure compatibility with evaluate library.
    
    This fixes compatibility issues where evaluate library expects DownloadConfig to have
    a 'token' attribute, but it might not be properly initialized in some versions.
    """
    # Patch 1: Ensure __init__ always sets token attribute
    original_init = DownloadConfig.__init__
    
    def patched_init(self, *args, **kwargs):
        """
        Patched __init__ that ensures token is always set.
        
        Handles compatibility issue where newer versions of datasets library
        may pass 'token' as a keyword argument, but the original __init__ 
        might not accept it. We remove 'token' from kwargs before calling
        original_init, then set it as an attribute afterward if needed.
        """
        # Remove 'token' from kwargs to avoid TypeError if original_init doesn't accept it
        # We'll set it as an attribute after initialization if it was provided
        token_value = kwargs.pop('token', None)
        
        # Call original __init__ without token in kwargs
        original_init(self, *args, **kwargs)
        
        # Set token as attribute if it was provided or if attribute doesn't exist
        if token_value is not None:
            self.token = token_value
        elif not hasattr(self, 'token'):
            self.token = None
    
    # Patch 2: Ensure copy() preserves token attribute
    original_copy = DownloadConfig.copy
    
    def patched_copy(self):
        """Patched copy method that ensures token is preserved"""
        copied = original_copy(self)
        # Ensure token attribute exists in copied instance
        if not hasattr(copied, 'token'):
            copied.token = getattr(self, 'token', None)
        return copied
    
    # Only patch if not already patched
    if not hasattr(DownloadConfig.__init__, '_patched'):
        DownloadConfig.__init__ = patched_init
        DownloadConfig.__init__._patched = True
    
    if not hasattr(DownloadConfig.copy, '_patched'):
        DownloadConfig.copy = patched_copy
        DownloadConfig.copy._patched = True


def _load_metric_safely(metric_name, experiment_id=None):
    """
    Load metric with fallback for compatibility issues between evaluate and datasets versions.
    
    This function handles the AttributeError where DownloadConfig object doesn't have 'token' attribute,
    which can occur due to version incompatibilities between evaluate and datasets libraries.
    
    Args:
        metric_name: Name of the metric to load (e.g., "accuracy", "squad")
        experiment_id: Unique identifier for this experiment to avoid cache collisions
                      between parallel evaluation instances (e.g., from multiple SLURM jobs)
    
    Returns:
        Loaded metric object
    """
    # Apply compatibility patch before loading
    _patch_download_config_for_compatibility()
    
    try:
        # Try loading with default configuration first, including experiment_id if provided
        if experiment_id is not None:
            return load(metric_name, experiment_id=experiment_id)
        else:
            return load(metric_name)
    except (AttributeError, TypeError) as e:
        error_msg = str(e)
        # Check if error is related to DownloadConfig.token attribute
        if "'DownloadConfig' object has no attribute 'token'" in error_msg or "token" in error_msg.lower():
            # Fallback 1: Try loading with download_config=None to avoid token issue
            try:
                if experiment_id is not None:
                    return load(metric_name, download_config=None, experiment_id=experiment_id)
                else:
                    return load(metric_name, download_config=None)
            except Exception as e2:
                # Fallback 2: Try with a custom DownloadConfig that explicitly handles token
                try:
                    download_config = DownloadConfig()
                    # Ensure token is set to None if attribute exists but causes issues
                    if hasattr(download_config, 'token'):
                        # Try to set token to None if possible
                        try:
                            download_config.token = None
                        except (AttributeError, TypeError):
                            pass
                    if experiment_id is not None:
                        return load(metric_name, download_config=download_config, experiment_id=experiment_id)
                    else:
                        return load(metric_name, download_config=download_config)
                except Exception as e3:
                    # Fallback 3: Try loading without any download_config parameter
                    # This might work if the library can create its own config
                    try:
                        if experiment_id is not None:
                            return load(metric_name, experiment_id=experiment_id)
                        else:
                            return load(metric_name)
                    except Exception:
                        # If all else fails, raise the original error with context
                        raise RuntimeError(
                            f"Failed to load metric '{metric_name}' due to compatibility issues. "
                            f"Original error: {error_msg}. "
                            f"Please check evaluate and datasets library versions."
                        ) from e
        else:
            # Re-raise if it's a different error
            raise


class Scorer(object):
    def __init__(self, metrics, experiment_id=None):
        """
        Initialize Scorer with metrics.
        
        Args:
            metrics: List of metric names to compute
            experiment_id: Unique identifier for this experiment to avoid cache collisions
                          between parallel evaluation instances (e.g., from multiple SLURM jobs)
        """
        self.metrics_toCompute = {"accuracy": False, "squad": False}

        if "Accuracy" in metrics:
            self.metrics_toCompute["accuracy"] = True
            # Use safe loading to handle version incompatibility issues
            self.accuracy_metric = _load_metric_safely("accuracy", experiment_id=experiment_id)

        if "Squad" in metrics:
            self.metrics_toCompute["squad"] = True
            # Use safe loading to handle version incompatibility issues
            self.squad_metric = _load_metric_safely("squad", experiment_id=experiment_id)

    def add_batch(self, batchOf_evalInfo):
        """
        Add batch to scorer

        Args:
            batchOf_evalInfo:

        Returns:

        """
        if self.metrics_toCompute["accuracy"]:
            self.accuracy_metric.add_batch(
                predictions=batchOf_evalInfo["predicted_choice"],
                references=batchOf_evalInfo["lbl"],
            )

        if self.metrics_toCompute["squad"]:
            self.squad_metric.add_batch(
                predictions=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "prediction_text": batchOf_evalInfo["prediction_text"],
                    }
                ),
                references=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "answers": batchOf_evalInfo["answers"],
                    }
                ),
            )

    def get_score(self):

        score = {}

        if self.metrics_toCompute["accuracy"]:
            score.update(self.accuracy_metric.compute())

        if self.metrics_toCompute["squad"]:
            squad_metrics = self.squad_metric.compute()
            # Scale SQUAD metrics to be between 0 and 1
            for metric, value in squad_metrics.items():
                squad_metrics[metric] = value / 100
            score.update(squad_metrics)

        for (key, value) in score.items():
            score[key] = float("%.3f" % value)

        score["average"] = get_average(score.values())

        return score
