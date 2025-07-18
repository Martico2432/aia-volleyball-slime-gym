import os
import numpy as np
from rlgym_ppo.util import MetricsLogger
from slime_api.slimestate import VolleyballState


class MyLogger(MetricsLogger):
    def _collect_metrics(self, game_state) -> list:
        game_state: VolleyballState = game_state
        player_0_id = next(iter(game_state.slimes))
        player_0_slime = game_state.slimes[player_0_id]


        point_scored = game_state.point_scored


        return [
                player_0_slime.touches_remaining,
                int(point_scored),
               ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_ball_touches_remaining = 0
        points = 0

        for metric in collected_metrics:
            avg_ball_touches_remaining += metric[0]
            points += metric[1]

        avg_ball_touches_remaining /= len(collected_metrics)

        report = {
            "average ball touches remaining": avg_ball_touches_remaining,
            "points": points,
        }

        wandb_run.log(report, step=cumulative_timesteps)


def build_indiedev_500_env():
    import gym
    from rlgym.api import RLGym
    from slime_api.slimeengine import IndieDevEngine
    from slime_api.slimeactions import SlimeActions
    from slime_api.slimeterminalcondition import IndieDevTerminalCondition
    from slime_api.slimetrucatedcondition import IndieDevTruncatedCondition
    from slime_api.slimeexampleobs import IndieDevDefaultObs
    from slime_api.slimemutator import IndieDevMutator
    from slime_api.slimerenderer import SlimeRenderer


    from rlgym.rocket_league.done_conditions import AnyCondition, AllCondition # Some condition that may be helpful
    from rlgym.rocket_league.action_parsers import  RepeatAction # Skip some actions, so faster training
    from rlgym.rocket_league.reward_functions import CombinedReward # Combine multiple rewards into one
    from rlgym.rocket_league.state_mutators import MutatorSequence # Combine multiple state mutators into one, and they will be applied in order


    from martico_rewards import PointRward, TouchesReward, BallDistanceReward




    import numpy as np
    from rlgym_ppo.util import RLGymV2GymWrapper




    action_parser = SlimeActions()
    termination_condition = IndieDevTerminalCondition()
    truncated_condition = IndieDevTruncatedCondition(600)
    state_mutator = IndieDevMutator()

    reward_fn = CombinedReward((PointRward(), 50), # 50 to prevent farming having 3 touches
                               (TouchesReward(), 0.2),
                               (BallDistanceReward(), 1)
                               )

    obs_builder = IndieDevDefaultObs()

    indie_dev_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncated_condition,
        transition_engine=IndieDevEngine(),
        renderer=SlimeRenderer("human"))

    wrapped_env = RLGymV2GymWrapper(indie_dev_env)
    wrapped_env.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)  # Set the action space to continuous throttle and steering, not automaticly set

    return wrapped_env  # Return the wrapped environment

if __name__ == "__main__":
    from rlgym_ppo import Learner
    n_proc = 32
    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    
    learner = Learner(build_indiedev_500_env, # DO NOT ADD () HERE, it will break the code
                      n_proc=n_proc, # number of processes to use for training
                      min_inference_size=min_inference_size, # IDK WHAT THIS IS
                      metrics_logger=MyLogger(), # METRICS LOGGER, None by default, will just make it log simple metrics
                      ppo_batch_size=50_000, # batch size - set this number to as large as your GPU can handle
                      policy_layer_sizes=[96, 96, 96], # policy network
                      critic_layer_sizes=[96, 96, 96], # value network
                      ts_per_iteration=50_000, # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=150_000, # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000, # minibatch size - set this less than or equal to the batch size
                      ppo_ent_coef=0.01, # entropy coefficient - this determines the impact of exploration on the policy #! I DONT KNOW HOW THIS WORKS WITH CONTINUOUS, but it should work
                      policy_lr=5e-4, # policy learning rate
                      critic_lr=5e-4, # value function learning rate
                      ppo_epochs=2,   # number of PPO epochs, recomended to keep this 2 or 3, but 2 can be faster than 3, 1 also works, but it learns slower
                      standardize_returns=True, # Idk what this does
                      standardize_obs=False, # Idk what this does
                      save_every_ts=1_000_000, # save every 1M steps
                      timestep_limit=1_000_000_000, # Train for 1B steps
                      wandb_project_name="slime_ai", # WandB project name
                      log_to_wandb=True,
                      render=True, #? Set to what you want, NOTE: rendering slows down 1 env if render delay is set to something, that is recomended to be 0.1
                      )
    

    
    print("Starting training...")
    learner.learn()