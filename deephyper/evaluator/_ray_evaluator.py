import logging
import os
import time
import traceback
from collections import defaultdict, deque, namedtuple

import ray
from deephyper.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)

NODES_PER_TASK = int(os.environ.get("DH_NODES_PER_TASK", 4))

def nodelist():
    """
    Get all compute nodes allocated in the current job context
    """
    node_str = os.environ["COBALT_PARTNAME"]
    # string like: 1001-1005,1030,1034-1200
    node_ids = []
    ranges = node_str.split(",")
    lo = None
    hi = None
    for node_range in ranges:
        lo, *hi = node_range.split("-")
        lo = int(lo)
        if hi:
            hi = int(hi[0])
            node_ids.extend(list(range(lo, hi + 1)))
        else:
            node_ids.append(lo)

    return [f"nid{node_id:05d}" for node_id in node_ids]

class RayFuture:
    FAIL_RETURN_VALUE = Evaluator.FAIL_RETURN_VALUE

    def __init__(self, func, x, nodes_queue):
        self.compute_objective = func
        self.nodes_queue = nodes_queue
        self.nodes = [self.nodes_queue.pop() for _ in range(NODES_PER_TASK)]
        self.id_res = self.compute_objective.remote(x, self.nodes)
        self._state = "active"
        self._result = None

    def _poll(self):
        if not self._state == "active":
            return

        id_done, _ = ray.wait([self.id_res], num_returns=1, timeout=0.001)

        if len(id_done) == 1:
            try:
                self._result = ray.get(id_done[0])
                self._state = "done"
            except Exception:
                print(traceback.format_exc())
                self._state = "failed"
            self.nodes_queue.extend(self.nodes)
        else:
            self._state = "active"

    def result(self):
        if not self.done:
            self._result = self.FAIL_RETURN_VALUE
        return self._result

    def cancel(self):
        pass  # NOT AVAILABLE YET

    @property
    def active(self):
        self._poll()
        return self._state == "active"

    @property
    def done(self):
        self._poll()
        return self._state == "done"

    @property
    def failed(self):
        self._poll()
        return self._state == "failed"

    @property
    def cancelled(self):
        self._poll()
        return self._state == "cancelled"


class RayEvaluator(Evaluator):
    """The RayEvaluator relies on the Ray (https://ray.readthedocs.io) package. Ray is a fast and simple framework for building and running distributed applications.

    Args:
        redis_address (str, optional): The "IP:PORT" redis address for the RAY-driver to connect on the RAY-head.
    """

    WaitResult = namedtuple("WaitResult", ["active", "done", "failed", "cancelled"])

    def __init__(
        self,
        run_function,
        cache_key=None,
        ray_address="auto",
        ray_password="5241590000000000",
        driver_num_cpus=None,
        driver_num_gpus=None,
        num_cpus_per_task=1,
        num_gpus_per_task=None,
        **kwargs,
    ):
        super().__init__(run_function, cache_key, **kwargs)

        logger.info(f"RAY Evaluator init: redis-address={ray_address}")

        if not ray_address is None:
            ray_init_kwargs = {}
            if driver_num_cpus:
                ray_init_kwargs["num_cpus"] = int(driver_num_cpus)
            if driver_num_gpus:
                ray_init_kwargs["num_gpus"] = int(driver_num_gpus)
            proc_info = ray.init(
                address=ray_address,
                _redis_password=ray_password,
                ignore_reinit_error=True,
            )
        else:
            proc_info = ray.init(ignore_reinit_error=True)

        self.num_cpus_per_tasks = num_cpus_per_task
        self.num_gpus_per_tasks = num_gpus_per_task

        self.num_cpus = int(
            sum([node["Resources"].get("CPU", 0) for node in ray.nodes()])
        )
        self.num_gpus = int(
            sum([node["Resources"].get("GPU", 0) for node in ray.nodes()])
        )
        self.num_workers = int(self.num_cpus // self.num_cpus_per_tasks)

        logger.info(
            f"RAY Evaluator will execute: '{self._run_function}', proc_info: {proc_info}"
        )

        self._run_function = ray.remote(
            num_cpus=self.num_cpus_per_tasks,
            num_gpus=self.num_gpus_per_tasks,
        )(self._run_function)
        self.nodes_queue = self.init_queue()

    def init_queue(self):
        return deque(nodelist())

    def _eval_exec(self, x: dict):
        assert isinstance(x, dict)
        future = RayFuture(self._run_function, x, self.nodes_queue)
        return future

    @staticmethod
    def _timer(timeout):
        if timeout is None:
            return lambda: True
        else:
            timeout = max(float(timeout), 0.01)
            start = time.time()
            return lambda: (time.time() - start) < timeout

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        assert return_when.strip() in ["ANY_COMPLETED", "ALL_COMPLETED"]
        waitall = bool(return_when.strip() == "ALL_COMPLETED")

        num_futures = len(futures)
        active_futures = [f for f in futures if f.active]
        time_isLeft = self._timer(timeout)

        if waitall:

            def can_exit():
                return len(active_futures) == 0

        else:

            def can_exit():
                return len(active_futures) < num_futures

        while time_isLeft():
            if can_exit():
                break
            else:
                active_futures = [f for f in futures if f.active]
                time.sleep(0.04)

        if not can_exit():
            raise TimeoutError(
                f"{timeout} sec timeout expired while "
                f"waiting on {len(futures)} tasks until {return_when}"
            )

        results = defaultdict(list)
        for f in futures:
            results[f._state].append(f)
        return self.WaitResult(
            active=results["active"],
            done=results["done"],
            failed=results["failed"],
            cancelled=results["cancelled"],
        )
