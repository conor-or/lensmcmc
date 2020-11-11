from tools import RunParser
from tools.obsfit import ObservationFit
from schwimmbad import MPIPool

# Get cmd line arguments
args = RunParser().parse_args_()

# Initialise MPI pool if needed
if args.processing:
    pool = MPIPool()
else:
    pool = None

# Start run
obsfit = ObservationFit(args.dir,
                        observation_settings_file=args.observation,
                        model_settings_file=args.model, pool=pool)

# Run sampler
obsfit.fit_model(pool=pool)
