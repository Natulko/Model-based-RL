#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth


def run_repetition(
        env,
        agent,
        n_timesteps: int,
        max_episode_length: int,
        eval_interval: int,
        **kwargs
) -> np.ndarray:
    rewards_arr = []
    s = env.reset()
    counter = 1
    for n_timestep in range(n_timesteps):
        a = agent.select_action(s, kwargs["epsilon"])
        s_next, r, done = env.step(a)
        agent.update(s, a, r, done, s_next, kwargs["n_planning_updates"])

        if n_timestep % eval_interval == 0:
            eval_env = WindyGridworld(wind_proportion=kwargs["wind_prop"])
            rewards_arr.append(agent.evaluate(eval_env))

        if done or counter == max_episode_length:
            s = env.reset()
            counter = 1
        else:
            s = s_next
            counter += 1

    env.reset()
    return np.array(rewards_arr)


def run_repetitions(
        env,
        agent_type: str,
        n_repetitions: int,
        n_timesteps: int,
        max_episode_length: int,
        eval_interval: int,
        **kwargs
) -> np.ndarray:
    curve = np.zeros((n_timesteps - 1) // eval_interval + 1)
    for _ in range(n_repetitions):
        # instantiate the agent
        if agent_type == "Dyna":
            agent = DynaAgent(
                n_states=env.n_states,
                n_actions=env.n_actions,
                learning_rate=kwargs["learning_rate"],
                gamma=kwargs["gamma"]
            )
        else:
            agent = PrioritizedSweepingAgent(
                n_states=env.n_states,
                n_actions=env.n_actions,
                learning_rate=kwargs["learning_rate"],
                gamma=kwargs["gamma"]
            )
        # run episodes
        repetition_curve = run_repetition(
            env,
            agent,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            **kwargs
        )
        curve += repetition_curve

    return curve / n_repetitions


def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    window_size = 8

    wind_proportions = [0.9, 1.0]
    n_planning_updates_arr = [1, 3, 5]

    # Assignment 1 - Dyna
    print("Dyna")
    dyna_best_n_plan_updt_arr = np.full(len(wind_proportions), n_planning_updates_arr[0])
    dyna_best_aucs = np.full(len(wind_proportions), -np.inf)
    for i in range(len(wind_proportions)):
        wind_prop = wind_proportions[i]
        env = WindyGridworld(wind_proportion=wind_prop)
        Dyna_plot = LearningCurvePlot(title=f"Dyna learning curves with {wind_prop} chance of wind")
        Q_curve = run_repetitions(
            env=env,
            agent_type="Dyna",
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            n_planning_updates=0,
            epsilon=epsilon,
            wind_prop=wind_prop
        )
        Dyna_plot.add_curve(range(len(Q_curve)), smooth(Q_curve, window_size),
                            label='Q-learning curve (baseline)')
        for n_planning_updates in n_planning_updates_arr:
            curve = run_repetitions(
                env=env,
                agent_type="Dyna",
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                max_episode_length=max_episode_length,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                n_planning_updates=n_planning_updates,
                epsilon=epsilon,
                wind_prop=wind_prop
            )
            Dyna_plot.add_curve(range(len(curve)), smooth(curve, window_size),
                                label=f"n_planning_updates={n_planning_updates}")
            curve_auc = np.trapz(curve)
            if curve_auc > dyna_best_aucs[i]:
                dyna_best_n_plan_updt_arr[i] = n_planning_updates
                dyna_best_aucs[i] = curve_auc

        Dyna_plot.save(f"Dyna_w{i}")

    # Assignment 2 - Prioritized Sweeping
    print("PS")
    ps_best_n_plan_updt_arr = np.full(len(wind_proportions), n_planning_updates_arr[0])
    ps_best_aucs = np.full(len(wind_proportions), -np.inf)
    for i in range(len(wind_proportions)):
        wind_prop = wind_proportions[i]
        env = WindyGridworld(wind_proportion=wind_prop)
        PS_plot = LearningCurvePlot(title=f"PS learning curves with {wind_prop} chance of wind")
        Q_curve = run_repetitions(
            env=env,
            agent_type="Dyna",
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            n_planning_updates=0,
            epsilon=epsilon,
            wind_prop=wind_prop
        )
        PS_plot.add_curve(range(len(Q_curve)), smooth(Q_curve, window_size),
                          label='Q-learning curve (baseline)')
        for n_planning_updates in n_planning_updates_arr:
            curve = run_repetitions(
                env=env,
                agent_type="PS",
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                max_episode_length=max_episode_length,
                eval_interval=eval_interval,
                learning_rate=learning_rate,
                gamma=gamma,
                n_planning_updates=n_planning_updates,
                epsilon=epsilon,
                wind_prop=wind_prop
            )
            PS_plot.add_curve(range(len(curve)), smooth(curve, window_size),
                              label=f"n_planning_updates={n_planning_updates}")
            curve_auc = np.trapz(curve)
            if curve_auc > ps_best_aucs[i]:
                ps_best_n_plan_updt_arr[i] = n_planning_updates
                ps_best_aucs[i] = curve_auc

        PS_plot.save(f"PS_w{i}")

    # Saving best params just in case
    # best_params = {
    #     "wind": wind_proportions,
    #     "Dyna_n_planning_updates": dyna_best_n_plan_updt_arr.tolist(),
    #     "PS_n_planning_updates": ps_best_n_plan_updt_arr.tolist()
    # }
    # with open("best_params.json", "w+") as file:
    #     json.dump(best_params, file)

    # Assignment 3 - Comparison
    print("Comparison")
    for i in range(len(wind_proportions)):
        wind_prop = wind_proportions[i]
        env = WindyGridworld(wind_proportion=wind_prop)
        comparison_plot = LearningCurvePlot(title=f"Comparison plot with {wind_prop} chance of wind")
        Q_curve = run_repetitions(
            env=env,
            agent_type="Dyna",
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            n_planning_updates=0,
            epsilon=epsilon,
            wind_prop=wind_prop
        )
        comparison_plot.add_curve(range(len(Q_curve)), smooth(Q_curve, window_size),
                                  label='Q-learning curve (baseline)')
        Dyna_curve = run_repetitions(
            env=env,
            agent_type="Dyna",
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            n_planning_updates=dyna_best_n_plan_updt_arr[i],
            epsilon=epsilon,
            wind_prop=wind_prop
        )
        comparison_plot.add_curve(
            range(len(Dyna_curve)), smooth(Dyna_curve, window_size),
            label=f'Dyna learning curve (n_pln_updt={dyna_best_n_plan_updt_arr[i]})'
        )
        PS_curve = run_repetitions(
            env=env,
            agent_type="PS",
            n_repetitions=n_repetitions,
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            gamma=gamma,
            n_planning_updates=ps_best_n_plan_updt_arr[i],
            epsilon=epsilon,
            wind_prop=wind_prop
        )
        comparison_plot.add_curve(
            range(len(PS_curve)), smooth(PS_curve, window_size),
            label=f'PS learning curve (n_pln_updt={dyna_best_n_plan_updt_arr[i]})'
        )

        comparison_plot.save(f"comparison_w{i}")


if __name__ == '__main__':
    experiment()
