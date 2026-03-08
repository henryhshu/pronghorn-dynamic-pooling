from . import ColdStartStrategy, FixedStrategy, RequestCentricStrategy, DynamicSystemStrategy
from . import Checkpoint, Parameters
import json
import os
from minio import Minio
import numpy as np


def cr_deserialize(payload: str, client: Minio):
    if not payload:
        # TODO FOR INTEGRATION: sensible defaults
        strategy_env = os.getenv("ENV").split(",")[0]
        print(f"Using strategy: {strategy_env}")
        if strategy_env == "cold":
            return ColdStartStrategy(Parameters(), [])
        elif strategy_env == "fixed&request_to_checkpoint=1":
            return FixedStrategy(Parameters(), [], 1)
        elif strategy_env == "dynamic_system":
            return DynamicSystemStrategy(Parameters(), [])
        else:
            return RequestCentricStrategy(Parameters(), [])
    obj = json.loads(payload)
    workload = Parameters.deserialize(obj["workload"])
    pool = [Checkpoint.deserialize(chkpt, client) for chkpt in obj["pool"]]
    print("Deserialized pool: ", pool)
    strategy = obj["strategy"]
    if strategy == "ColdStart":
        return ColdStartStrategy(workload, pool)
    elif strategy == "Fixed":
        return FixedStrategy(workload, pool, obj["request_to_checkpoint"])
    elif strategy == "RequestCentric":
        strategy = RequestCentricStrategy(
            workload,
            pool,
            obj["max_capacity"],
            obj["p"],
            obj["gamma"],
            eps=obj["eps"],
        )
        strategy.weights = np.array(obj["weights"])
        return strategy
    elif strategy == "DynamicSystem":
        strat = DynamicSystemStrategy(
            workload,
            pool,
            max_pool_size=obj["max_pool_size"],
            min_pool_size=obj["min_pool_size"],
            base_pool_size=obj["base_pool_size"],
            p=obj["p"],
            gamma=obj["gamma"],
            eps=obj["eps"],
            low_var_thresh=obj["low_var_thresh"],
            high_var_thresh=obj["high_var_thresh"],
            variance_window=obj["variance_window"],
        )
        strat.weights = np.array(obj["weights"])
        strat._recent_latencies = obj["recent_latencies"]
        strat._effective_capacity = obj["effective_capacity"]
        return strat
