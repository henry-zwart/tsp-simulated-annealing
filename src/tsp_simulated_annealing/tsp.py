from dataclasses import dataclass

import numpy as np

from tsp_simulated_annealing.acceptance import acceptance
from tsp_simulated_annealing.cooling_schedules import Cooling, get_scheduler
from tsp_simulated_annealing.data import ProblemData
from tsp_simulated_annealing.main import distance_route, two_opt


@dataclass
class TemperatureTuneResult:
    initial: float
    initial_ci: float
    final: float
    final_ci: float


def average_increase(
    s0: np.ndarray,
    locations: np.ndarray,
    iters: int,
    repeats: int,
) -> np.ndarray:
    avg_increases = []
    for _ in range(repeats):
        increases = []
        state = s0
        dist = distance_route(s0, locations)
        for _ in range(iters):
            new_state = two_opt(state, 125)
            new_dist = distance_route(new_state, locations)

            if new_dist > dist:
                increases.append(new_dist - dist)

            # Unconditionally update the state
            state = new_state
            dist = new_dist
        avg_increases.append(sum(increases) / len(increases))
    return np.array(avg_increases)


def tune_temperature(
    s0: np.ndarray,
    problem: ProblemData,
    init_accept: float,
    final_accept: float,
    warmup_iters_init: int = 100,
    warmup_repeats_init: int = 100,
    warmup_iters_final: int = 10,
    warmup_repeats_final: int = 1000,
) -> TemperatureTuneResult:
    """
    Calculate the initial temperature which yields a desired initial accept probability.

    Runs 2-Opt for a number of iterations, accepting all uphill steps, and recording
    the average increase across uphill steps. The initial temperature is calculated as:

        T_0 = (Average increase)/ln(p)

    Where p is the desired initial acceptance probability.
    """
    avg_incr_init = average_increase(
        s0, problem.locations, warmup_iters_init, warmup_repeats_init
    )
    init_temp = -avg_incr_init / np.log(init_accept)

    avg_incr_final = average_increase(
        problem.optimal_tour,
        problem.locations,
        warmup_iters_final,
        warmup_repeats_final,
    )
    final_temp = -avg_incr_final / np.log(final_accept)

    return TemperatureTuneResult(
        initial=init_temp.mean(),
        final=final_temp.mean(),
        initial_ci=1.97 * init_temp.std(ddof=1) / np.sqrt(warmup_repeats_init),
        final_ci=1.97 * final_temp.std(ddof=1) / np.sqrt(warmup_repeats_final),
    )


def solve_tsp(
    s0: np.ndarray,
    problem: ProblemData,
    cooling_algo: Cooling,
    cool_time: int,
    iters_per_temp: int = 1,
    init_temp: float | None = None,
    final_temp: float | None = None,
    init_accept: float = 0.95,
    final_accept: float = 0.01,
    warmup_iters_init: int = 100,
    warmup_repeats_init: int = 100,
    warmup_iters_final: int = 100,
    warmup_repeats_final: int = 100,
) -> np.ndarray:
    """
    Solve a TSP problem with simulated annealing.

    First estimates parameters for cooling function:
        - Initial temperature, using warmup samples to determine value which
            yields desired initial acceptance ratio
        - Cooling-schedule specific parameters.

    Then runs simulated annealing to solve the TSP problem using the 2-Opt
    operation. Simulation is terminated at max_samples, or when the temperature
    drops below a specified threshold.
    """
    if init_temp is None or final_temp is None:
        temp_tune_results = tune_temperature(
            s0,
            problem,
            init_accept,
            final_accept,
            warmup_iters_init,
            warmup_repeats_init,
            warmup_iters_final,
            warmup_repeats_final,
        )
        init_temp = init_temp or temp_tune_results.initial
        final_temp = final_temp or temp_tune_results.final
        final_temp = init_temp / 1000

    update_temperature = get_scheduler(init_temp, final_temp, cool_time, cooling_algo)

    # Data records
    states = []

    temperature = init_temp
    state = s0
    dist = problem.distance(state)
    for time in range(cool_time):
        for _ in range(iters_per_temp):
            new_state = two_opt(state, 125)
            new_dist = problem.distance(new_state)
            # states.append(new_state)

            alpha = acceptance(new_dist, dist, temperature)
            if new_dist < dist or np.random.uniform(0, 1) < alpha:
                state = new_state
                dist = new_dist

        states.append(state)
        temperature = update_temperature(time)
        if temperature < final_temp:
            break

    return np.array(states)
