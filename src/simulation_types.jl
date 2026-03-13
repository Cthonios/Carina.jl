# Simulation container types for YAML-driven single-domain simulations.

import FiniteElementContainers as FEC

# ---------------------------------------------------------------------------
# Single-domain simulation
# ---------------------------------------------------------------------------

"""
    SingleDomainSimulation

Holds all objects needed to run one domain: FEC parameters, a time
integrator, a post-processor, and bookkeeping for output.
"""
struct SingleDomainSimulation{Params, Integrator, PP}
    params          ::Params        # FEC.Parameters
    integrator      ::Integrator    # QuasiStaticIntegrator | NewmarkIntegrator
    post_processor  ::PP            # FEC.PostProcessor
    n_steps         ::Int
    output_interval ::Int           # write output every N steps
    use_gpu         ::Bool          # true when params/integrator live on the GPU
end
