import argparse
import tomllib
import signal
import shutil
import uuid
import ast
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich.console import Console
from rich.traceback import install

import clean_pufferl
import environment
import policy

# Rich tracebacks
install(show_locals=False)

# Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args["policy"])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args["rnn"])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args["train"]["device"])


def init_wandb(args, name, id=None, resume=True):
    import wandb

    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args["wandb_project"],
        group=args["wandb_group"],
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=args,
        name=name,
    )
    return wandb


def sweep(args, env_name, make_env, policy_cls, rnn_cls):
    import wandb

    sweep_id = wandb.sweep(sweep=args["sweep"], project=args["wandb_project"])

    def main():
        try:
            wandb = init_wandb(args, env_name, id=args["exp_id"])
            args["train"].update(wandb.config.train)
            train(args, make_env, policy_cls, rnn_cls, wandb)
        except Exception as e:  # noqa
            Console().print_exception()

    wandb.agent(sweep_id, main, count=100)


### CARBS Sweeps
def sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls):
    from math import log, ceil, floor
    import numpy as np

    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam

    # from carbs import ParamDictType
    from carbs import Param

    def closest_power(x):
        possible_results = floor(log(x, 2)), ceil(log(x, 2))
        return int(2 ** min(possible_results, key=lambda z: abs(x - 2**z)))

    def carbs_param(
        group,
        name,
        space,
        wandb_params,
        mmin=None,
        mmax=None,
        search_center=None,
        is_integer=False,
        rounding_factor=1,
    ):
        wandb_param = wandb_params[group]["parameters"][name]
        if "values" in wandb_param:
            values = wandb_param["values"]
            mmin = min(values)
            mmax = max(values)

        if mmin is None:
            mmin = float(wandb_param["min"])
        if mmax is None:
            mmax = float(wandb_param["max"])

        if space == "log":
            Space = LogSpace
            if search_center is None:
                search_center = 2 ** (np.log2(mmin) + np.log2(mmax) / 2)
        elif space == "linear":
            Space = LinearSpace
            if search_center is None:
                search_center = (mmin + mmax) / 2
        elif space == "logit":
            Space = LogitSpace
            assert mmin == 0
            assert mmax == 1
            assert search_center is not None
        else:
            raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

        return Param(
            name=f"{group}-{name}",
            space=Space(min=mmin, max=mmax, is_integer=is_integer, rounding_factor=rounding_factor),
            search_center=search_center,
        )

    if not os.path.exists("checkpoints"):
        os.system("mkdir checkpoints")

    import wandb

    sweep_id = wandb.sweep(
        sweep=args["sweep"],
        project="carbs",
    )
    target_metric = args["sweep"]["metric"]["name"].split("/")[-1]
    sweep_parameters = args["sweep"]["parameters"]
    # wandb_env_params = sweep_parameters['env']['parameters']
    # wandb_policy_params = sweep_parameters['policy']['parameters']

    # Must be hardcoded and match wandb sweep space for now
    param_spaces = []
    if "total_timesteps" in sweep_parameters["train"]["parameters"]:
        time_param = sweep_parameters["train"]["parameters"]["total_timesteps"]
        min_timesteps = time_param["min"]
        param_spaces.append(
            carbs_param(
                "train",
                "total_timesteps",
                "log",
                sweep_parameters,
                search_center=min_timesteps,
                is_integer=True,
            )
        )

    batch_param = sweep_parameters["train"]["parameters"]["batch_size"]
    default_batch = (batch_param["max"] - batch_param["min"]) // 2

    minibatch_param = sweep_parameters["train"]["parameters"]["minibatch_size"]
    default_minibatch = (minibatch_param["max"] - minibatch_param["min"]) // 2

    if "env" in sweep_parameters:
        env_params = sweep_parameters["env"]["parameters"]

        # MOBA
        if "reward_death" in env_params:
            param_spaces.append(
                carbs_param("env", "reward_death", "linear", sweep_parameters, search_center=-1.0)
            )
        if "reward_xp" in env_params:
            param_spaces.append(
                carbs_param("env", "reward_xp", "linear", sweep_parameters, search_center=0.006)
            )
        if "reward_distance" in env_params:
            param_spaces.append(
                carbs_param(
                    "env", "reward_distance", "linear", sweep_parameters, search_center=0.05
                )
            )
        if "reward_tower" in env_params:
            param_spaces.append(
                carbs_param("env", "reward_tower", "linear", sweep_parameters, search_center=3.0)
            )

        # Atari
        if "frameskip" in env_params:
            param_spaces.append(
                carbs_param(
                    "env", "frameskip", "linear", sweep_parameters, search_center=4, is_integer=True
                )
            )
        if "repeat_action_probability" in env_params:
            param_spaces.append(
                carbs_param(
                    "env",
                    "repeat_action_probability",
                    "logit",
                    sweep_parameters,
                    search_center=0.25,
                )
            )

    param_spaces += [
        # carbs_param('cnn_channels', 'linear', wandb_policy_params, search_center=32, is_integer=True),
        # carbs_param('hidden_size', 'linear', wandb_policy_params, search_center=128, is_integer=True),
        # carbs_param('vision', 'linear', search_center=5, is_integer=True),
        carbs_param("train", "learning_rate", "log", sweep_parameters, search_center=1e-3),
        carbs_param("train", "gamma", "logit", sweep_parameters, search_center=0.95),
        carbs_param("train", "gae_lambda", "logit", sweep_parameters, search_center=0.90),
        carbs_param(
            "train", "update_epochs", "linear", sweep_parameters, search_center=1, is_integer=True
        ),
        carbs_param("train", "clip_coef", "logit", sweep_parameters, search_center=0.1),
        carbs_param("train", "vf_coef", "logit", sweep_parameters, search_center=0.5),
        carbs_param("train", "vf_clip_coef", "logit", sweep_parameters, search_center=0.1),
        carbs_param("train", "max_grad_norm", "linear", sweep_parameters, search_center=0.5),
        carbs_param("train", "ent_coef", "log", sweep_parameters, search_center=0.01),
        carbs_param(
            "train",
            "batch_size",
            "log",
            sweep_parameters,
            search_center=default_batch,
            is_integer=True,
        ),
        carbs_param(
            "train",
            "minibatch_size",
            "log",
            sweep_parameters,
            search_center=default_minibatch,
            is_integer=True,
        ),
        carbs_param(
            "train", "bptt_horizon", "log", sweep_parameters, search_center=16, is_integer=True
        ),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )
    carbs = CARBS(carbs_params, param_spaces)

    def main():
        wandb = init_wandb(args, env_name, id=args["exp_id"])
        wandb.config.__dict__["_locked"] = {}
        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print("Suggestion:", suggestion)
        # cnn_channels = suggestion.pop('cnn_channels')
        # hidden_size = suggestion.pop('hidden_size')
        # vision = suggestion.pop('vision')
        # wandb.config.env['vision'] = vision
        # wandb.config.policy['cnn_channels'] = cnn_channels
        # wandb.config.policy['hidden_size'] = hidden_size
        train_suggestion = {
            k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("train-")
        }
        env_suggestion = {k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("env-")}
        args["train"].update(train_suggestion)
        args["train"]["batch_size"] = closest_power(train_suggestion["batch_size"])
        args["train"]["minibatch_size"] = closest_power(train_suggestion["minibatch_size"])
        args["train"]["bptt_horizon"] = closest_power(train_suggestion["bptt_horizon"])

        args["env"].update(env_suggestion)
        args["track"] = True
        wandb.config.update({"train": args["train"]}, allow_val_change=True)

        # args.env.__dict__['vision'] = vision
        # args['policy']['cnn_channels'] = cnn_channels
        # args['policy']['hidden_size'] = hidden_size
        # args['rnn']['input_size'] = hidden_size
        # args['rnn']['hidden_size'] = hidden_size
        print(wandb.config.train)
        print(wandb.config.env)
        print(wandb.config.policy)
        try:
            stats, uptime = train(args, make_env, policy_cls, rnn_cls, wandb)
        except Exception as e:  # noqa
            is_failure = True  # noqa
            import traceback

            traceback.print_exc()
        else:
            observed_value = [s[target_metric] for s in stats if target_metric in s]
            if len(observed_value) > 0:
                observed_value = np.mean(observed_value)
            else:
                observed_value = 0

            print(f"Observed value: {observed_value}")
            obs_out = carbs.observe(  # noqa
                ObservationInParam(
                    input=orig_suggestion,
                    output=observed_value,
                    cost=uptime,
                )
            )

    wandb.agent(sweep_id, main, count=500)


def train(args, make_env, policy_cls, rnn_cls, wandb):
    if args["vec"] == "serial":
        vec = pufferlib.vector.Serial
    elif args["vec"] == "multiprocessing":
        vec = pufferlib.vector.Multiprocessing
    else:
        raise ValueError("Invalid --vector (serial/multiprocessing).")

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=args["env"],
        num_envs=args["train"]["num_envs"],
        num_workers=args["train"]["num_workers"],
        batch_size=args["train"]["env_batch_size"],
        zero_copy=args["train"]["zero_copy"],
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    train_config = pufferlib.namespace(
        **args["train"],
        env=env_name,
        exp_id=args["exp_id"] or env_name + "-" + str(uuid.uuid4())[:8],
    )
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    uptime = data.profile.uptime
    stats = []
    for _ in range(10):  # extra data for sweeps
        stats.append(clean_pufferl.evaluate(data)[0])

    clean_pufferl.close(data)
    return stats, uptime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments for myosuite", add_help=False)
    parser.add_argument("-c", "--config", default="config.toml")
    parser.add_argument(
        "-e",
        "--env-name",
        type=str,
        default="myoElbowPose1D6MRandom-v0",
        help="Name of specific environment to run",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices="train eval evaluate sweep sweep-carbs autotune p5rofile".split(),
    )
    parser.add_argument("--eval-model-path", type=str, default=None)
    parser.add_argument(
        "--baseline", action="store_true", help="Pretrained baseline where available"
    )
    parser.add_argument(
        "--vec",
        "--vector",
        "--vectorization",
        type=str,
        default="serial",
        choices=["serial", "multiprocessing"],
    )
    parser.add_argument(
        "--exp-id", "--exp-name", type=str, default=None, help="Resume from experiment"
    )
    parser.add_argument("--wandb-project", type=str, default="myosuite")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    args = parser.parse_known_args()[0]

    # Load config file
    if not os.path.exists(args.config):
        raise Exception(f"Config file {args.config} not found")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    for section in config:
        for key in config[section]:
            argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=config[section][key])

    # Override config with command line arguments
    parsed = parser.parse_args().__dict__
    args = {"env": {}, "policy": {}, "rnn": {}}
    env_name = parsed.pop("env_name")
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:  # noqa
            prev[subkey] = value

    # Load env binding and policy
    make_env = environment.env_creator(env_name)
    policy_cls = getattr(policy, args["base"]["policy_name"])
    rnn_cls = None
    if "rnn_name" in args["base"]:
        rnn_cls = getattr(policy, args["base"]["rnn_name"])

    # Process mode
    if args["baseline"]:
        assert args["mode"] in ("train", "eval", "evaluate")
        args["track"] = True
        version = ".".join(pufferlib.__version__.split(".")[:2])
        args["exp_id"] = f"puf-{version}-{env_name}"
        args["wandb_group"] = f"puf-{version}-baseline"
        shutil.rmtree(f'experiments/{args["exp_id"]}', ignore_errors=True)
        run = init_wandb(args, env_name, args["exp_id"], resume=False)
        if args["mode"] in ("eval", "evaluate"):
            model_name = f"puf-{version}-{env_name}_model:latest"
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args["eval_model_path"] = os.path.join(data_dir, model_file)
    if args["mode"] == "train":
        wandb = None
        if args["track"]:
            wandb = init_wandb(args, env_name, id=args["exp_id"])
        train(args, make_env, policy_cls, rnn_cls, wandb=wandb)
    elif args["mode"] in ("eval", "evaluate"):
        clean_pufferl.rollout(
            make_env,
            args["env"],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            model_path=args["eval_model_path"],
            device=args["train"]["device"],
        )
    elif args["mode"] == "sweep":
        args["track"] = True
        sweep(args, env_name, make_env, policy_cls, rnn_cls)
    elif args["mode"] == "sweep-carbs":
        sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls)
    elif args["mode"] == "autotune":
        pufferlib.vector.autotune(make_env, batch_size=args["train"]["env_batch_size"])
    elif args["mode"] == "profile":
        import cProfile

        cProfile.run("train(args, env_module, make_env)", "stats.profile")
        import pstats
        from pstats import SortKey

        p = pstats.Stats("stats.profile")
        p.sort_stats(SortKey.TIME).print_stats(10)
