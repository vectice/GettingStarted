from __future__ import annotations

import inspect

from typing import Any, Dict
from vectice.models.validation import TestSuiteReturnType


## You just pass your function as an argument
def Vectice_wrapper_function(
    module: callable,
    internal_functions_param: Dict[str, Any],
) -> TestSuiteReturnType:
    
    # Inspect the signature of the internal function
    signature = inspect.signature(module)

    # Validate that all required parameters are provided
    for param_name, param in signature.parameters.items():
        if param.default == inspect.Parameter.empty and param_name not in internal_functions_param:
            raise ValueError(f"Missing required parameter: {param_name}")

    # Filter out any extra parameters not in the signature
    filtered_params = {param_name: internal_functions_param[param_name] for param_name in signature.parameters if param_name in internal_functions_param}

    # Run the provided callable with filtered parameters
    result = module(**filtered_params)

    # Helper function to extract paths
    def extract_paths(obj):
        paths = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                paths.extend(extract_paths(value))
        elif isinstance(obj, list):
            for item in obj:
                paths.extend(extract_paths(item))
        elif isinstance(obj, str):
            paths.append(obj)
        elif hasattr(obj, 'attachments'):
            paths.extend(extract_paths(obj.attachments))
        return paths

    # Extract paths from the result
    extracted_paths = extract_paths(result)

    # Convert the result to a dictionary
    output_files = {
        "paths": extracted_paths,
    }

    # Return in the expected format
    return TestSuiteReturnType(**output_files)



def Vectice_wrapper(
    output_files: Dict[str, Any] = {"paths": None, "dataframes": None, "metrics": None, "properties": None},
) -> TestSuiteReturnType:

    ####
    #####Paste your code Here
    ##### 
    


    # RETURN IN THE VECTICE EXPECTED FORMART
    return TestSuiteReturnType(
        metrics=output_files["metrics"],
        properties=output_files["properties"],
        tables=output_files["dataframes"],
        attachments=output_files["paths"],
    )
