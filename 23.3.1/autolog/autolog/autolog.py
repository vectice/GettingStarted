# This captures the code called, we then use regex to identify the variables in locals. The variables found can then be logged to vectice.
from __future__ import annotations
import copy
import re
import IPython
from pandas import DataFrame
from IPython import get_ipython


ip = IPython.get_ipython()


def cell_vars(offset=1):
    import io
    from contextlib import redirect_stdout

    ipy = get_ipython()
    out = io.StringIO()

    with redirect_stdout(out):
        ipy.magic("history {0}".format(ipy.execution_count - offset))

    # process each line...
    x = out.getvalue().replace(" ", "").split("\n")
    x = [a.split("=")[0] for a in x if "=" in a]  # all of the variables in the cell
    g = globals()
    result = {k: g[k] for k in x if k in g}
    return result


def _get_local_variables():
    cell_content = ip.get_parent()['content']['code']
    cell_info = ip.get_parent()
    return cell_content, cell_info




def _check_registred_event(ip):
    # This will unregister our event and prevent stacking 
    ip.events.unregister('post_run_cell', _get_local_variables)


def _identify_assets(local_vars):
    cell_content, cell_info = _get_local_variables()
    matches = re.findall(r'(.*?) =', cell_content)
    vectice_data = {variable: local_vars[variable] for variable in matches}

    sklearn_model = get_model(vectice_data)

    pandas_df = get_df(vectice_data)

    train_test = get_x_y(vectice_data)

    model_metrics = _get_model_metrics(vectice_data, cell_content)

    graph = get_graph(local_vars)

    return sklearn_model, pandas_df, train_test, model_metrics, graph


def get_x_y(vectice_data):
    from numpy import ndarray
    train_test = {}
    for key in vectice_data.keys():
        if isinstance(vectice_data[key], ndarray):
            train_test[key] = copy.deepcopy(vectice_data[key])
    return train_test


def get_model(vectice_data):
    for key in vectice_data.keys():
        if "sklearn" and "model" in str(vectice_data[key].__class__):
            return copy.deepcopy(vectice_data[key])
    return None


def get_df(vectice_data):
    for key in vectice_data.keys():
        if isinstance(vectice_data[key], DataFrame):
            return copy.deepcopy(vectice_data[key])
    return None


def _format_model(model, model_metrics):
    from vectice import Model

    params = model.get_params()
    algorithm = str(model.__class__).split(".")[-1]

    return Model(library="sklearn", technique=algorithm, metrics=model_metrics, properties=params, predictor=model)


def _get_model_metrics(vectice_data, cell_content):
    # check the variable instantiation
    from sklearn.metrics import _regression
    regression_functions = dir(_regression)
    metrics = []
    metrics_return = {}
    for func in regression_functions:
        for func in regression_functions:
            metric = re.findall(r"(.*?) = mean_absolute_error", cell_content)
            if metric:
                metrics += metric
    for key in vectice_data.keys():
        if key in metrics:
            metrics_return[key] = vectice_data[key]
    return metrics_return


def _get_arrays(train_test):
    return train_test["x"], train_test["y"]


def get_graph(local_vars):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    for key in local_vars:
        if isinstance(local_vars[key], Figure):
            return local_vars[key]
        if plt is local_vars[key]:
            return local_vars[key]
    return None


def _log_to_vectice(model = None, dataset = None, train_test = None, model_metrics = None, graph = None):
    import vectice

    vec = vectice.connect(
        api_token="qrA92ZLgV.zEKa357LJ62WXeyn9GmlqrA92ZLgVBod0k1ZQ8wbRMNp4xvYPD", host="qa.vectice.com")  # Paste your API token

    phase = vec.phase("PHA-6660")  # Paste your own Modeling Phase ID

    model_iteration = phase.create_iteration()

    # if train_test:
    #     x_array, y_array = _get_arrays(train_test)

    if model:
        model = _format_model(model, model_metrics)
        model_iteration.log(model)
    if dataset is not None:
        from vectice import Dataset
        vec_dataset = Dataset.clean(vectice.FileResource("dummy.csv", dataframes=dataset))
        model_iteration.log(vec_dataset)

    if graph:
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()

        graph.savefig(rf"{temp_dir.name}\test.png")
        model_iteration.log(rf"{temp_dir.name}\test.png")


def autolog(local_vars):
    # from . import vw
    # ip.events.register('post_run_cell', _get_local_variables)
    model, dataset, train_test, model_metrics, graph = _identify_assets(local_vars)
    _log_to_vectice(model, dataset, train_test, model_metrics, graph)



