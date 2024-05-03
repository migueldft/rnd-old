import numpy as np
from loguru import logger
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args


class Optimizer(type):
    _dim_types = {
        "Real": Real,
        "Integer": Integer,
        "Categorical": Categorical,
    }

    @classmethod
    def instantiate_spec(cls, name, spec):
        if isinstance(spec, dict):
            dim_type = cls._dim_types[spec["dim_type"]]
            kwargs = {k: v for (k, v) in spec.items() if k != "dim_type"}
            return dim_type(name=name, **kwargs)
        else:
            return Categorical([spec], name=name)

    @classmethod
    def hyperparameter_space_factory(cls, specs):
        return [cls.instantiate_spec(name, spec) for name, spec in specs.items()]

    @classmethod
    def callback_factory(cls, num_steps):
        def optimization_callback(res):
            current_value = -res.func_vals[-1]
            best_value = -res.func_vals.min()
            current_step = len(res.func_vals)
            logger.info(
                f"Optimization {current_step:5d}/{num_steps:5d}. "
                f"Current value: {current_value:5.3f}. "
                f"Best value: {best_value:5.3f}."
            )

        return optimization_callback

    @classmethod
    def fix_type(cls, x):
        if not hasattr(x, "dtype"):
            return x
        elif np.issubdtype(x.dtype, np.int_):
            return int(x)
        elif np.issubdtype(x.dtype, np.float_):
            return float(x)

    @classmethod
    def minimize(cls, loss, hyperspec, n_opt_steps, n_random_starts, random_seed):
        hyperparameter_space = cls.hyperparameter_space_factory(hyperspec)
        _loss = use_named_args(hyperparameter_space)(loss)
        _get_params = use_named_args(hyperparameter_space)(lambda **kw: kw)
        opt_results = gp_minimize(
            _loss,
            hyperparameter_space,
            n_random_starts=n_random_starts,
            n_calls=n_opt_steps + n_random_starts,
            random_state=random_seed,
            callback=cls.callback_factory(n_opt_steps + n_random_starts),
        )
        return _get_params([cls.fix_type(value) for value in opt_results.x])
