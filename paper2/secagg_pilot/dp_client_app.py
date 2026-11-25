"""Client app for the separate central-DP plus SecAgg+ pilot."""
from flwr.client import ClientApp
from flwr.client.mod import fixedclipping_mod, secaggplus_mod

from paper2.secagg_pilot.client_app import client_fn


# Match Flower 1.42's official combined DP+SecAgg example exactly.  The list
# order is significant: SecAgg+ is registered first and fixed clipping second.
app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod, fixedclipping_mod])
