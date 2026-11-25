"""Flower ServerApp for the no-evaluation Paper 2 SecAgg+ pilot."""
from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow

from paper2.restart.flower_secagg import SecAggPilotConfig, build_workflow
from paper2.secagg_pilot.task import fit_config, initial_parameters, state_to_ndarrays


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    settings = SecAggPilotConfig()
    features = int(context.run_config["features"])
    initial = ndarrays_to_parameters(state_to_ndarrays(initial_parameters(features)))
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=settings.participants,
        min_available_clients=settings.participants,
        fraction_evaluate=0.0,
        initial_parameters=initial,
        on_fit_config_fn=fit_config,
    )
    legacy = LegacyContext(context=context, config=ServerConfig(num_rounds=int(context.run_config["num-server-rounds"])), strategy=strategy)
    DefaultWorkflow(fit_workflow=build_workflow(settings))(grid, legacy)
