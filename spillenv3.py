"""
Optimal Stopping Environment for CO2 Injection in Region 2.

Reframes the CCS problem:
  - Inject ONLY in Region 2 (fixed rate per step)
  - Agent decides WHEN to stop, WHEN to measure gravity
  - Two objectives (scalarized into dollars):
      1. (trapped_R2 - spillover) in Mt × $70M/Mt
      2. Cost of time-lapse gravity measurement: −$30,000 per measurement

Actions (Discrete(4)):
  0 = Stop                       → episode terminates, reward = 0
  1 = Inject + Measure gravity   → inject CO2, pay for measurement
  2 = Inject only                → inject CO2, no measurement
  3 = Measure only               → no injection, pay for measurement

Observation (32-dim):
  [0:30]  Last measured gravity vector (µGal); zeros until first measurement
  [30]    Cumulative injected volume (normalised by R2 capacity)
  [31]    Steps since last gravity measurement (normalised)

Episode ends when:
  - Agent selects action 0 (stop)
  - Safety truncation at max_steps (default 500)

Spillover detection from gravity:
  Sensors above Regions 1 and 3 read ≈0 when CO2 is confined to R2.
  Non-zero readings outside R2 indicate spillover — the agent must
  learn this spatial pattern from the 30-element gravity vector.

Usage:
    from env_optimal_stopping import OptimalStoppingCO2Env
    env = OptimalStoppingCO2Env(fixed_seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # inject + measure
"""

import numpy as np
from typing import Optional, Tuple, Dict, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is available.")

from spillenv2 import build_valid_env


class OptimalStoppingCO2Env(gym.Env):
    """
    Optimal-stopping POMDP for CO2 injection in Region 2.
    See module docstring for full description.
    """

    metadata = {"render_modes": ["human"]}

    # Human-readable action names
    ACTION_NAMES = {
        0: "STOP",
        1: "INJECT + MEASURE",
        2: "INJECT (no measure)",
        3: "MEASURE (no inject)",
    }

    def __init__(
        self,
        fixed_seed: int = 42,
        nx_sensors: int = 30,
        nz_sensors: int = 30,
        nx_cells: int = 100,
        nz_cells: int = 100,
        obs_well_loc_m: float = 10000.0,
        injection_rate_m3: float = 999_999.99,
        max_steps: int = 500,
        co2_price_per_Mt: float = 70.0,       # M$ per million tonnes trapped
        measurement_cost_M: float = 0.03,      # M$ per measurement ($30 000)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.fixed_seed = fixed_seed
        self.nx_sensors = nx_sensors
        self.injection_rate_m3 = injection_rate_m3
        self.max_steps = max_steps
        self.render_mode = render_mode

        # ---- Economic parameters (all in M$) ----
        self.co2_price_per_Mt = co2_price_per_Mt
        self.measurement_cost_M = measurement_cost_M
        self.rho_co2 = 650.0  # kg/m³

        # ---- Build spillpoint simulation (fixed geology) ----
        print(f"[OptimalStoppingCO2Env] Building env with seed {fixed_seed} ...")
        self.sim_env = build_valid_env(
            start_seed=fixed_seed,
            obs_well_loc=obs_well_loc_m,
            nx_cells=nx_cells,
            nz_cells=nz_cells,
            nx_sensors=nx_sensors,
            nz_sensors=nz_sensors,
        )
        self.scaler = self.sim_env.scaler

        # Region 2 injection site
        self.inject_loc_sim = self.sim_env.region_injector_locs_sim[2]
        self.injection_vol_sim = self.scaler.to_sim_vol(injection_rate_m3)

        # Region 2 capacity (for observation normalisation)
        self.r2_capacity_m3 = self.scaler.to_phys_vol(self.sim_env.cap2)
        print(f"  Region 2 capacity: {self.r2_capacity_m3/1e6:.2f} Mm³")
        print(f"  Injection rate:    {injection_rate_m3/1e6:.2f} Mm³/step")
        print(f"  ≈ steps to fill R2: {self.r2_capacity_m3 / injection_rate_m3:.0f}")

        # Spillpoint boundaries (for analysis / plotting)
        self.xv1_phys = self.scaler.to_phys_x(self.sim_env.xv1)
        self.xv3_phys = self.scaler.to_phys_x(self.sim_env.xv3)

        # ---- Gym spaces ----
        self.action_space = spaces.Discrete(4)
        # obs = [gravity(30), cumulative_injected_norm(1), steps_since_measure_norm(1)]
        obs_dim = nx_sensors + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.obs_dim = obs_dim

        # ---- Episode state (set in reset) ----
        self._init_episode_state()

    # ==================================================================
    # Internal state helpers
    # ==================================================================
    def _init_episode_state(self):
        self.current_step = 0
        self.cumulative_injected_m3 = 0.0
        self.cumulative_trapped_r2_m3 = 0.0
        self.cumulative_spillover_m3 = 0.0
        self.cumulative_leaked_m3 = 0.0
        self.cumulative_reward = 0.0
        self.last_gravity = np.zeros(self.nx_sensors, dtype=np.float32)
        self.steps_since_measure = 0
        self.num_measurements = 0
        self.has_spilled = False

    def _reset_fluid(self):
        """Reset CO2 in the simulator, keeping geology fixed."""
        if hasattr(self.sim_env, "reset_fluid_state"):
            self.sim_env.reset_fluid_state()
        else:
            sim = self.sim_env
            sim.v_reg = {1: 0.0, 2: 0.0, 3: 0.0}
            sim.v_shared_12 = 0.0
            sim.v_shared_23 = 0.0
            sim.v_shared_123 = 0.0
            sim.v_leaked = 0.0

    def _m3_to_Mt(self, volume_m3: float) -> float:
        """Convert m³ of CO2 to millions of tonnes."""
        return volume_m3 * self.rho_co2 / 1e9

    # ==================================================================
    # Gym API
    # ==================================================================
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_fluid()
        self._init_episode_state()

        obs = self._get_obs()
        info = self._build_info(action=None, delta_trapped_r2_m3=0.0,
                                delta_spillover_m3=0.0, measured=False)
        return obs, info

    def step(self, action: int):
        """
        Execute one step.

        Returns: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # ---- Action 0: STOP ----
        if action == 0:
            obs = self._get_obs()
            info = self._build_info(action=0, delta_trapped_r2_m3=0.0,
                                    delta_spillover_m3=0.0, measured=False)
            return obs, 0.0, True, False, info   # terminated=True

        inject = action in (1, 2)
        measure = action in (1, 3)

        # ---- Injection ----
        delta_trapped_r2_m3 = 0.0
        delta_spillover_m3 = 0.0

        if inject:
            result = self.sim_env.inject(self.inject_loc_sim, self.injection_vol_sim)

            # CO2 that stays in Region 2
            trapped_r2_sim = result.get("to_r2", 0.0)
            delta_trapped_r2_m3 = self.scaler.to_phys_vol(trapped_r2_sim)

            # Spillover = everything NOT in R2  (R1, R3, shared zones, leaked)
            spill_keys = ["to_r1", "to_r3", "to_s12", "to_s23", "to_s123", "leaked"]
            spillover_sim = sum(result.get(k, 0.0) for k in spill_keys)
            delta_spillover_m3 = self.scaler.to_phys_vol(spillover_sim)

            self.cumulative_injected_m3 += self.injection_rate_m3
            self.cumulative_trapped_r2_m3 += delta_trapped_r2_m3
            self.cumulative_spillover_m3 += (
                delta_spillover_m3 - self.scaler.to_phys_vol(result.get("leaked", 0.0))
            )
            self.cumulative_leaked_m3 += self.scaler.to_phys_vol(result.get("leaked", 0.0))

            if delta_spillover_m3 > 1e-3:
                self.has_spilled = True

        # ---- Reward (M$) ----
        #   Obj 1: (trapped_R2 − spillover) converted to Mt × $70 M/Mt
        #   Obj 2: −$0.03 M if measured
        delta_trapped_Mt = self._m3_to_Mt(delta_trapped_r2_m3)
        delta_spillover_Mt = self._m3_to_Mt(delta_spillover_m3)
        reward = (delta_trapped_Mt - delta_spillover_Mt) * self.co2_price_per_Mt

        if measure:
            reward -= self.measurement_cost_M

        self.cumulative_reward += reward

        # ---- Gravity measurement ----
        if measure:
            _, grav_values = self.sim_env.get_time_lapse_gravity()
            self.last_gravity = np.array(grav_values, dtype=np.float32)
            self.steps_since_measure = 0
            self.num_measurements += 1
        else:
            self.steps_since_measure += 1

        # ---- Step counter & truncation ----
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._build_info(action=action,
                                delta_trapped_r2_m3=delta_trapped_r2_m3,
                                delta_spillover_m3=delta_spillover_m3,
                                measured=measure)
        return obs, float(reward), False, truncated, info

    # ==================================================================
    # Observation
    # ==================================================================
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        # [0:30]  last gravity measurement
        obs[: self.nx_sensors] = self.last_gravity
        # [30]    cumulative injected volume / R2 capacity  (clipped to [0, 5])
        obs[self.nx_sensors] = min(
            self.cumulative_injected_m3 / max(self.r2_capacity_m3, 1.0), 5.0
        )
        # [31]    steps since last measurement / 50  (clipped to [0, 1])
        obs[self.nx_sensors + 1] = min(self.steps_since_measure / 50.0, 1.0)
        return obs

    # ==================================================================
    # Info dict
    # ==================================================================
    def _build_info(self, action, delta_trapped_r2_m3, delta_spillover_m3,
                    measured) -> dict:
        return {
            "step": self.current_step,
            "action": action,
            "action_name": self.ACTION_NAMES.get(action, "N/A"),
            "delta_trapped_r2_m3": delta_trapped_r2_m3,
            "delta_spillover_m3": delta_spillover_m3,
            "cumulative_injected_m3": self.cumulative_injected_m3,
            "cumulative_trapped_r2_m3": self.cumulative_trapped_r2_m3,
            "cumulative_spillover_m3": self.cumulative_spillover_m3,
            "cumulative_leaked_m3": self.cumulative_leaked_m3,
            "cumulative_reward_M": self.cumulative_reward,
            "has_spilled": self.has_spilled,
            "measured": measured,
            "num_measurements": self.num_measurements,
            "r2_fill_fraction": self.cumulative_trapped_r2_m3 / max(self.r2_capacity_m3, 1.0),
        }

    # ==================================================================
    # Convenience
    # ==================================================================
    def get_action_meanings(self) -> List[str]:
        return [f"{i}: {name}" for i, name in self.ACTION_NAMES.items()]

    def get_sensor_positions_m(self) -> np.ndarray:
        """Physical x-positions (m) of the 30 surface gravity sensors."""
        return self.scaler.to_phys_x(self.sim_env.grav_sensor_x_sim)

    def get_region_boundaries_m(self) -> Tuple[float, float]:
        """Return (xv1_m, xv3_m) — physical positions of the two spillpoints."""
        return self.xv1_phys, self.xv3_phys

    def __repr__(self):
        return (
            f"OptimalStoppingCO2Env(seed={self.fixed_seed}, "
            f"R2_cap={self.r2_capacity_m3/1e6:.1f}Mm³, "
            f"inj_rate={self.injection_rate_m3/1e6:.1f}Mm³/step, "
            f"max_steps={self.max_steps})"
        )