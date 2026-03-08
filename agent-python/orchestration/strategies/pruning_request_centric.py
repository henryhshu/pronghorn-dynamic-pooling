import random
import numpy as np
from agent_python.orchestration.strategies.request_centric import RequestCentricStrategy

class PruningRequestCentricStrategy(RequestCentricStrategy):
    """
    Subclass of RequestCentricStrategy with custom pruning:
    Removes checkpoints with high avg latency and low variance.
    """
    def _prune_pool(self):
        output = []
        if len(self.pool) == 0:
            return
        avg_latencies = [c.avg_response_time for c in self.pool]
        var_latencies = [c.var_response_time for c in self.pool]
        avg_threshold = np.percentile(avg_latencies, 75)  # top 25% high latency
        var_threshold = np.percentile(var_latencies, 25)  # bottom 25% low variance
        to_remove = [
            c for c in self.pool
            if c.avg_response_time >= avg_threshold and c.var_response_time <= var_threshold
        ]
        for chkpt in to_remove:
            print(f"Pruning checkpoint with high avg ({chkpt.avg_response_time:.2f}) and low var ({chkpt.var_response_time:.2f})")
            chkpt.delete()
        # Keep the rest using original logic
        by_performance = sorted(
            [c for c in self.pool if c not in to_remove],
            key=lambda c: self._weights_for(c.state.request_number, scalar=True),
            reverse=True,
        )
        keeping_p = round(self.p * len(by_performance))
        output += by_performance[:keeping_p]
        by_performance = by_performance[keeping_p:]
        keeping_gamma = round(self.gamma * len(by_performance))
        output += random.choices(
            by_performance, k=min(keeping_gamma, len(by_performance))
        )
        output_chkpts = {chkpt for chkpt in output}
        removed = [chkpt for chkpt in self.pool if chkpt not in output_chkpts and chkpt not in to_remove]
        for chkpt in removed:
            chkpt.delete()
        self.pool[:] = output
        print(
            f"Evicted all but top {keeping_p} by performance and {keeping_gamma} by random, plus {len(to_remove)} by high-latency-low-variance rule"
        )
        assert len(self.pool) <= self.max_capacity
