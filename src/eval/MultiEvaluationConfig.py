import itertools

from src.eval.EvaluationConfig import EvaluationConfig
from src.utils.utils import breadth_first_search

class MultiEvaluationConfig(EvaluationConfig):
    def __init__(self,
                 fields_toIterateOver,
                 values_toIterateOver,
                 configDict_toInitializeFrom,
                 fields_toUpdate=None,
                 kwargs=None):
        '''

        Args:
            fields_toIterateOver: list of fields to iterate over
            conditionalFieldValues_toIterateOver: nested dictionary of possible values for 1
                                                  field to possible values for another field
                                                  where the possible values for each field depend on
                                                  one another
                                                  - Must have a mapping fields_toIterateOver to a
                                                  list of fields to iterate over where each field in
                                                  the list is the field of the
                                                  corresponding keys in that dictionary
            configDict_toInitializeFrom:
            fields_toUpdate:
            kwargs:
        '''
        super().__init__(configDict_toInitializeFrom, fields_toUpdate, kwargs)

        self.fields_toIterateOver = fields_toIterateOver
        self.values_toIterateOver = values_toIterateOver

    def get_allConfigs(self):
        '''

        Returns:

        '''

        iterated_configs = []

        if self.values_toIterateOver is None:
            listOf_listOfValues_toIterateOver = [self.get_dict()[k] for k in self.fields_toIterateOver]
            all_valueSettings = list(itertools.product(*listOf_listOfValues_toIterateOver))
        else:
            all_valueSettings = breadth_first_search(self.values_toIterateOver)

        # Get base_dict once before the loop (more efficient)
        base_dict = self.get_dict()
        
        for value_setting in all_valueSettings:
            updated_fields = dict(zip(self.fields_toIterateOver, value_setting))
            
            # CRITICAL: Always preserve prediction_dir from base config
            # This prevents NoneType errors in getAndMake_specificPredictionDir
            # We must ALWAYS set prediction_dir in updated_fields, even if fields_toIterateOver
            # contains prediction_dir (which would set it to None from value_setting)
            # Priority 1: Use prediction_dir from self (may have been set by fields_toUpdate in __init__)
            if hasattr(self, 'prediction_dir') and self.prediction_dir is not None:
                updated_fields["prediction_dir"] = self.prediction_dir
            # Priority 2: Use prediction_dir from base_dict if self.prediction_dir is None
            elif "prediction_dir" in base_dict and base_dict["prediction_dir"] is not None:
                updated_fields["prediction_dir"] = base_dict["prediction_dir"]
            # Priority 3: If still None, raise error (should not happen if evaluate_checkpoint is correct)
            else:
                raise ValueError(
                    f"prediction_dir is None in MultiEvaluationConfig.get_allConfigs(). "
                    f"This should not happen. self.prediction_dir={getattr(self, 'prediction_dir', 'N/A')}, "
                    f"base_dict['prediction_dir']={base_dict.get('prediction_dir', 'N/A')}"
                )
            
            # Final check: ensure prediction_dir is not None in updated_fields (defensive programming)
            if updated_fields.get("prediction_dir") is None:
                raise ValueError(
                    f"prediction_dir is None in updated_fields after setting it. "
                    f"This should not happen. self.prediction_dir={getattr(self, 'prediction_dir', 'N/A')}, "
                    f"base_dict['prediction_dir']={base_dict.get('prediction_dir', 'N/A')}"
                )

            new_config = EvaluationConfig(
                configDict_toInitializeFrom=base_dict,
                fields_toUpdate=updated_fields
            )

            iterated_configs.append(new_config)

        return iterated_configs