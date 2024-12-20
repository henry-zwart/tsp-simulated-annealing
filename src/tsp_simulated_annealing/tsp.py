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


@dataclass
class OptimisationResult:
    initial_state: np.ndarray
    initial_temp: float
    final_temp: float
    temperature_steps: int
    states: np.ndarray


def average_increase(
    s0: np.ndarray,
    locations: np.ndarray,
    iters: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    avg_increases = []
    for _ in range(repeats):
        increases = []
        state = s0
        new_state = state.copy()
        dist = distance_route(s0, locations)
        for _ in range(iters):
            new_state, dist_delta = two_opt(state, locations, rng)
            new_dist = dist + dist_delta

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
    rng: np.random.Generator,
    warmup_iters: int = 100,
    warmup_repeats: int = 100,
) -> TemperatureTuneResult:
    """
    Calculate the initial temperature which yields a desired initial accept probability.

    Runs 2-Opt for a number of iterations, accepting all uphill steps, and recording
    the average increase across uphill steps. The initial temperature is calculated as:

        T_0 = (Average increase)/ln(p)

    Where p is the desired initial acceptance probability.
    """
    avg_incr_init = average_increase(
        s0, problem.locations, warmup_iters, warmup_repeats, rng
    )
    init_temp = -avg_incr_init / np.log(init_accept)

    final_temp = init_temp / 1000

    return TemperatureTuneResult(
        initial=init_temp.mean(),
        final=final_temp.mean(),
        initial_ci=1.97 * init_temp.std(ddof=1) / np.sqrt(warmup_repeats),
        final_ci=1.97 * final_temp.std(ddof=1) / np.sqrt(warmup_repeats),
    )


def solve_tsp(
    s0: np.ndarray,
    problem: ProblemData,
    cooling_algo: Cooling,
    cool_time: int,
    rng: np.random.Generator,
    iters_per_temp: int = 1,
    init_temp: float | None = None,
    final_temp: float | None = None,
    init_accept: float = 0.95,
    warmup_iters: int = 100,
    warmup_repeats: int = 100,
) -> OptimisationResult:
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
        print("Tuning temperature")
        temp_tune_results = tune_temperature(
            s0,
            problem,
            init_accept,
            rng,
            warmup_iters,
            warmup_repeats,
        )
        init_temp = init_temp or temp_tune_results.initial
        final_temp = final_temp or temp_tune_results.final
        final_temp = init_temp / 1000

    update_temperature = get_scheduler(init_temp, final_temp, cool_time, cooling_algo)

    # Data records
    states = np.zeros((cool_time + 1, len(s0)), dtype=np.int64)
    states[0] = s0

    temperature = init_temp
    state = s0
    new_state = state.copy()
    dist = problem.distance(state)
    for time in range(cool_time + 1):
        states[time] = state
        temperature = update_temperature(time)

        for _ in range(iters_per_temp):
            new_state, dist_delta = two_opt(state, problem.locations, rng)
            new_dist = dist + dist_delta

            alpha = acceptance(new_dist, dist, temperature)
            if new_dist < dist or rng.uniform(0, 1) < alpha:
                state = new_state
                dist = new_dist

    return OptimisationResult(
        initial_state=s0,
        initial_temp=init_temp,
        final_temp=final_temp,
        temperature_steps=cool_time + 1,
        states=states,
    )


def sample_at_temperature(
    s0: np.ndarray,
    problem: ProblemData,
    n_samples: int,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from the Markov chain at fixed temperature."""
    # Data records
    states = np.zeros((n_samples, len(s0)), dtype=np.int64)
    states[0] = s0

    state = s0
    new_state = state.copy()
    dist = problem.distance(state)
    for i in range(n_samples):
        new_state, dist_delta = two_opt(state, problem.locations, rng)
        new_dist = dist + dist_delta

        alpha = acceptance(new_dist, dist, temperature)
        if new_dist < dist or rng.uniform(0, 1) < alpha:
            state = new_state
            dist = new_dist
        states[i] = state

    return states
