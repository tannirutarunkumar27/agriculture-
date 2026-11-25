"""Flower ClientApp: local training plus the vetted SecAgg+ client modifier."""
from __future__ import annotations

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from paper2.secagg_pilot.task import initial_parameters, load_client_training_data, local_fit, set_ndarrays


class Paper2SecureClient(NumPyClient):
    def __init__(self, x, y, client_id: str, learning_rate: float):
        self.x, self.y, self.client_id, self.learning_rate = x, y, client_id, learning_rate
        self.model = initial_parameters(x.shape[1])

    def fit(self, parameters, config):
        set_ndarrays(self.model, parameters)
        values = local_fit(self.model, self.x, self.y, self.client_id, int(config["server_round"]), self.learning_rate)
        # Equal fixed-client weighting; sample counts are deliberately not released.
        return values, 1, {}

    def evaluate(self, parameters, config):
        raise RuntimeError("Pilot evaluation is disabled; no test or client metric release")


def client_fn(context: Context):
    node = context.node_config
    if "client-id" in node:
        client_id = node["client-id"]
    else:
        roster = context.run_config["client-roster"].split(",")
        partition_id = int(node["partition-id"])
        if partition_id < 0 or partition_id >= len(roster) or not roster[partition_id]:
            raise ValueError("Flower partition ID is outside the frozen client roster")
        client_id = roster[partition_id]
    x, y = load_client_training_data(context.run_config["data-root"], context.run_config["fold-id"], client_id)
    return Paper2SecureClient(x, y, client_id, float(context.run_config["learning-rate"])).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
