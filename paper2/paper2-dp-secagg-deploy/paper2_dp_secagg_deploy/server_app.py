"""Deployment ServerApp for the frozen 20-round central-DP plus SecAgg+ study."""
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow

from .protocol import SecAggProtocol, workflow
from .task import make_regressor, state_to_ndarrays


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = SecAggProtocol()
    run = context.run_config
    if int(run["num-server-rounds"]) != 20 or run.get("central-dp") is not True or run.get("differential-privacy-enabled") is not True:
        raise ValueError("Deployment run differs from the frozen 20-round central-DP protocol")
    if run.get("outer-test-evaluation") is not False:
        raise ValueError("Outer-test evaluation is prohibited")
    initial = ndarrays_to_parameters(state_to_ndarrays(make_regressor(int(run["features"]))))
    base = FedAvg(
        fraction_fit=1.0, min_fit_clients=config.participants,
        min_available_clients=config.participants, fraction_evaluate=0.0,
        initial_parameters=initial,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )
    strategy = DifferentialPrivacyClientSideFixedClipping(
        base, noise_multiplier=float(run["noise-multiplier"]), clipping_norm=float(run["clipping-norm"]),
        num_sampled_clients=config.participants,
    )
    legacy = LegacyContext(context=context, config=ServerConfig(num_rounds=20), strategy=strategy)
    DefaultWorkflow(fit_workflow=workflow(config))(grid, legacy)
