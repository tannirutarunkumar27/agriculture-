"""Server app for the separate central-DP plus SecAgg+ pilot.

Central DP protects the released aggregate under its stated assumptions; it
does not hide that aggregate from the trusted server.
"""
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow

from paper2.restart.flower_secagg import SecAggPilotConfig, build_workflow
from paper2.secagg_pilot.task import fit_config, initial_parameters, state_to_ndarrays


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    settings = SecAggPilotConfig()
    initial = ndarrays_to_parameters(state_to_ndarrays(initial_parameters(int(context.run_config["features"]))))
    base = FedAvg(
        fraction_fit=1.0, min_fit_clients=settings.participants,
        min_available_clients=settings.participants, fraction_evaluate=0.0,
        initial_parameters=initial,
        on_fit_config_fn=fit_config,
    )
    strategy = DifferentialPrivacyClientSideFixedClipping(
        base, noise_multiplier=float(context.run_config["noise-multiplier"]),
        clipping_norm=float(context.run_config["clipping-norm"]),
        num_sampled_clients=settings.participants,
    )
    legacy = LegacyContext(context=context, config=ServerConfig(num_rounds=int(context.run_config["num-server-rounds"])), strategy=strategy)
    DefaultWorkflow(fit_workflow=build_workflow(settings))(grid, legacy)
