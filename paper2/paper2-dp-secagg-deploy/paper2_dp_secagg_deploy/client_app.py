"""Deployment ClientApp: each SuperNode supplies its own local data location."""
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
from flwr.common import Context

from .task import load_training_data, make_regressor, set_ndarrays, train_one_round


class SecureClient(NumPyClient):
    def __init__(self, x, y, client_id: str, learning_rate: float):
        self.x, self.y, self.client_id, self.learning_rate = x, y, client_id, learning_rate
        self.model = make_regressor(x.shape[1])

    def fit(self, parameters, config):
        set_ndarrays(self.model, parameters)
        values = train_one_round(self.model, self.x, self.y, self.client_id, int(config["server_round"]), self.learning_rate)
        return values, 1, {}  # Equal fixed client weighting; no client metric release.

    def evaluate(self, parameters, config):
        raise RuntimeError("Evaluation is disabled: the deployment app never reads test targets")


def client_fn(context: Context):
    node = context.node_config
    required = ("client-id", "data-root", "fold-id")
    if any(not isinstance(node.get(name), str) or not node[name] for name in required):
        raise ValueError("Deployment SuperNode requires client-id, data-root, and fold-id node configuration")
    x, y = load_training_data(node["data-root"], node["fold-id"], node["client-id"])
    if x.shape[1] != int(context.run_config["features"]):
        raise ValueError("Client feature count differs from frozen server configuration")
    return SecureClient(x, y, node["client-id"], float(context.run_config["learning-rate"])).to_client()


# Modifier order is frozen from Flower's combined central-DP/SecAgg+ example.
app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod, fixedclipping_mod])
