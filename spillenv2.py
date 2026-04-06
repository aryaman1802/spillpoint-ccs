"""
Modified POMDP Carbon Storage Model - Multi-Region Injection (v2)

Key Changes from v1:
1. CONFIGURABLE OBSERVATION SPACE: Choose timelapse gravity, borehole gravity, or both
2. DISCRETE ACTION SPACE: Matches the predefined ACTIONS list (injection rate tuples)
3. REWARD: (trapped CO2 - leaked CO2) in Mm³ per step
4. FLEXIBLE LEAKAGE: Episode does NOT end on leakage; max_steps=150
5. OPTIMIZED GRAVITY CACHING: Reduces computation overhead
6. FIXED SEED: Same geology for ALL episodes
"""

import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Union, Dict
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Try to import tqdm, use a fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

print("Imports done!")

# ==============================================================================
# DISCRETE ACTION SPACE DEFINITION
# ==============================================================================
# These are predefined injection rate tuples (region1, region2, region3)
# Sum should be ~1,000,000 (total injection rate in m³/year distributed across regions)
DISCRETE_ACTIONS = [
    (0, 0, 999999.99), 
    (0, 333333.33, 666666.66), 
    (0, 666666.66, 333333.33), 
    (0, 999999.99, 0), 
    (333333.33, 0, 666666.66), 
    (333333.33, 333333.33, 333333.33), 
    (333333.33, 666666.66, 0), 
    (666666.66, 0, 333333.33), 
    (666666.66, 333333.33, 0), 
    (999999.99, 0, 0)
]
N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)

# ==============================================================================
# 1. GEOMETRY & PARAMETERS
# ==============================================================================

def h_top(x: np.ndarray, lobe: float, center: float, elev: float) -> np.ndarray:
    """Calculates the structural elevation profile of the reservoir top (Caprock)."""
    return lobe * np.sin(5 * np.pi * x) + center * np.sin(np.pi * x) + elev * x


@dataclass
class SubsurfaceParams:
    """Data Class for storing the random parameters."""
    lobe: float
    center: float
    elev: float
    rho: float


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generates random geological models."""
    lobes = rng.uniform(0.05, 0.25, size=n)
    centers = rng.uniform(0.05, 0.5, size=n)
    elevs = rng.uniform(0.05, 0.5, size=n)
    rhos = rng.uniform(0.5, 1.5, size=n)
    return np.array([SubsurfaceParams(a, b, c, d) for a, b, c, d in zip(lobes, centers, elevs, rhos)])


# ==============================================================================
# 2. SCALING HELPERS
# ==============================================================================

class PhysicalScaler:
    """Handles the conversion between Simulation Units and Physical Units."""
    
    def __init__(self, x_len_m: float = 20000.0, res_thickness_m: float = 500.0, 
                 slice_thickness_m: float = 5000.0, top_depth_m: float = 2000.0):
        self.x_scale = x_len_m
        self.y_scale = res_thickness_m
        self.z_thick = slice_thickness_m
        self.top_depth = top_depth_m
        self.base_depth = top_depth_m + res_thickness_m

    def to_sim_x(self, x_meters: float) -> float:
        return x_meters / self.x_scale

    def to_sim_vol(self, vol_m3: float) -> float:
        return vol_m3 / (self.x_scale * self.y_scale * self.z_thick)

    def to_phys_x(self, sim_x: float) -> float:
        return sim_x * self.x_scale

    def to_phys_vol(self, sim_vol: float) -> float:
        return sim_vol * (self.x_scale * self.y_scale * self.z_thick)

    def to_phys_depth(self, sim_h: float) -> float:
        return self.base_depth - (sim_h * self.y_scale)


# ==============================================================================
# 3. MAIN SIMULATION CLASS (SpillpointEnv)
# ==============================================================================

class SpillpointEnv:
    """The Core Simulation Engine."""
    
    def __init__(self, params: SubsurfaceParams,
                 obs_well_loc_m: float = 11000.0,
                 nx_cells: int = 100,
                 nz_cells: int = 100,
                 nx_sensors: int = 60,
                 nz_sensors: int = 50):
        self.params = params
        self.obs_well_loc_m = obs_well_loc_m

        self.nx_cells = nx_cells
        self.nz_cells = nz_cells
        self.nx_sensors = nx_sensors
        self.nz_sensors = nz_sensors
        
        # Initialize Scaler
        # Original: reservoir top at 2000 m depth (weaker gravity signal at surface):
        # self.scaler = PhysicalScaler(
        #     x_len_m=20000.0,
        #     res_thickness_m=500.0,
        #     slice_thickness_m=50.0,
        #     top_depth_m=2000.0
        # )
        # Shallow reservoir: top at 300 m depth (stronger gravity signal — sensors
        # are closer to CO2, so timelapse gravity better resolves anticline shape):
        # self.scaler = PhysicalScaler(
        #     x_len_m=20000.0,
        #     res_thickness_m=500.0,
        #     slice_thickness_m=50.0,
        #     top_depth_m=300.0
        # )
        self.scaler = PhysicalScaler(
            x_len_m=20000.0,
            res_thickness_m=500.0,
            slice_thickness_m=5000.0,
            top_depth_m=2000.0
        )
        
        # Generate Geometry
        self.x = np.linspace(0.0, 1.0, self.nx_cells)
        self.dx = self.x[1] - self.x[0]
        self.h = h_top(self.x, params.lobe, params.center, params.elev)

        self.h0 = float(self.h[0])
        self.h1 = float(self.h[-1])

        self.xv1, self.v1, self.xv3, self.v3 = self._find_two_minima()

        self.mask1 = self.x <= self.xv1
        self.mask2 = (self.x >= self.xv1) & (self.x <= self.xv3)
        self.mask3 = self.x >= self.xv3

        self.hmax1 = float(self.h[self.mask1].max())
        self.hmax2 = float(self.h[self.mask2].max())
        self.hmax3 = float(self.h[self.mask3].max())
        self.hmax_all = float(self.h.max())

        # Define Spill Thresholds
        self.thresh1 = max(self.v1, self.h0)
        self.thresh2 = max(self.v1, self.v3)
        self.thresh3 = max(self.v3, self.h1)

        self.top12, self.bot12 = self.v1, max(self.h0, self.v3)
        self.top23, self.bot23 = self.v3, max(self.h1, self.v1)
        self.top123, self.bot123 = min(self.v1, self.v3), max(self.h0, self.h1)

        # Initialize Fluid State
        self.v_reg = {1: 0.0, 2: 0.0, 3: 0.0}
        self.v_shared_12 = 0.0
        self.v_shared_23 = 0.0
        self.v_shared_123 = 0.0
        self.v_leaked = 0.0

        self.cap1 = self._volume_region_at_level(self.mask1, self.thresh1)
        self.cap2 = self._volume_region_at_level(self.mask2, self.thresh2)
        self.cap3 = self._volume_region_at_level(self.mask3, self.thresh3)
        self.cap12 = self._volume_between_levels(self.mask1 | self.mask2, self.top12, self.bot12)
        self.cap23 = self._volume_between_levels(self.mask2 | self.mask3, self.top23, self.bot23)
        self.cap123 = self._volume_between_levels(np.ones_like(self.x, bool), self.top123, self.bot123)

        # Gravity Physics Setup
        self.G = 6.67408e-11
        self.rho_brine = 1000.0
        self.rho_co2 = 650.0
        self.density_diff = self.rho_co2 - self.rho_brine
        self.si_to_ugal = 1e8
        
        self._init_gravity_grid(nz_cells=nz_cells, ns_sensors=nx_sensors)
        self._init_borehole_gravity(nz_sensors=nz_sensors)
        
        # Store injector locations for each region (center of each region)
        self.region_injector_locs_sim = self._compute_region_centers()

    def _compute_region_centers(self) -> Dict[int, float]:
        """Compute the center x-location for each of the 3 regions."""
        center1 = 0.5 * self.xv1
        center2 = 0.5 * (self.xv1 + self.xv3)
        center3 = 0.5 * (self.xv3 + 1.0)
        return {1: center1, 2: center2, 3: center3}

    def get_region_injector_locations_m(self) -> Dict[int, float]:
        """Get injector locations in physical units (meters)."""
        return {
            r: self.scaler.to_phys_x(loc) 
            for r, loc in self.region_injector_locs_sim.items()
        }

    # ------------------------ Part A: Geometry Solvers ------------------------

    def _find_two_minima(self) -> Tuple[float, float, float, float]:
        y = self.h
        idx = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
        idx = idx[(idx > 0) & (idx < len(y) - 1)]
        
        if len(idx) < 2:
            interior = np.arange(1, len(y) - 1)
            idx = interior[np.argsort(y[interior])[:2]]
            idx.sort()
        
        i1, i3 = int(idx[0]), int(idx[1])
        return float(self.x[i1]), float(y[i1]), float(self.x[i3]), float(y[i3])

    def _volume_region_at_level(self, mask: np.ndarray, level: float) -> float:
        dh = np.clip(self.h[mask] - level, 0.0, None)
        return float(self.params.rho * np.trapezoid(dh, self.x[mask]))

    def _volume_between_levels(self, mask: np.ndarray, top: float, bottom: float) -> float:
        if bottom >= top:
            return 0.0
        v_top = self._volume_region_at_level(mask, top)
        v_bottom = self._volume_region_at_level(mask, bottom)
        return float(v_bottom - v_top)

    def _invert_volume_to_level(self, mask: np.ndarray, lo: float, hi: float, target: float) -> float:
        V_lo = self._volume_region_at_level(mask, lo)
        V_hi = self._volume_region_at_level(mask, hi)
        target = float(np.clip(target, 0.0, V_lo))
        
        if abs(target - V_lo) < 1e-12:
            return lo
        if target <= V_hi + 1e-12:
            return hi
        
        a, b = lo, hi
        for _ in range(80):
            m = 0.5 * (a + b)
            Vm = self._volume_region_at_level(mask, m)
            if Vm > target:
                a = m
            else:
                b = m
            if abs(Vm - target) < 1e-10 or abs(b - a) < 1e-10:
                return m
        return 0.5 * (a + b)

    # ------------------------ Part B: Public API ------------------------

    def find_region(self, xloc: float) -> int:
        if xloc <= self.xv1:
            return 1
        if xloc < self.xv3:
            return 2
        return 3

    def calc_vol_rem(self, region: int) -> float:
        return [None, self.cap1 - self.v_reg[1], self.cap2 - self.v_reg[2], self.cap3 - self.v_reg[3]][region]

    def calc_vol_rem_shared_region(self, a: int, b: int) -> float:
        s = {a, b}
        if s == {1, 2}:
            return self.cap12 - self.v_shared_12
        if s == {2, 3}:
            return self.cap23 - self.v_shared_23
        return 0.0

    def calc_rem_vol_all_shared_region(self) -> float:
        return self.cap123 - self.v_shared_123

    # ------------------------ Part C: Fluid Injection Logic ------------------------
    
    def _next_after_full(self, region: int) -> tuple:
        if region == 1:
            return ('spill', 2) if (self.v1 > self.h0) else ('leak', None)
        if region == 3:
            return ('spill', 2) if (self.v3 > self.h1) else ('leak', None)
        
        if self.v1 < self.v3:
            return ('spill', 3)
        if self.v1 > self.v3:
            return ('spill', 1)
        if self.h0 < self.h1:
            return ('spill', 3)
        if self.h0 > self.h1:
            return ('spill', 1)
        rem1, rem3 = self.calc_vol_rem(1), self.calc_vol_rem(3)
        if rem1 < rem3:
            return ('spill', 3)
        if rem1 > rem3:
            return ('spill', 1)
        return ('spill', int(np.random.choice([1, 3])))

    def inject(self, xloc: float, volume: float) -> dict:
        """Inject CO2 at a specific location."""
        EPS = 1e-12
        cur = self.find_region(float(xloc))
        S = {"to_r1": 0.0, "to_r2": 0.0, "to_r3": 0.0, "to_s12": 0.0, "to_s23": 0.0, "to_s123": 0.0, "leaked": 0.0}

        rem_cur = max(0.0, self.calc_vol_rem(cur))
        add = min(volume, rem_cur)
        self.v_reg[cur] += add
        S[f"to_r{cur}"] += add
        left = volume - add
        if left <= EPS:
            return S

        mode, adj = self._next_after_full(cur)
        if mode == 'leak':
            self.v_leaked += left
            S["leaked"] += left
            return S

        rem_adj = max(0.0, self.calc_vol_rem(adj))
        add = min(left, rem_adj)
        self.v_reg[adj] += add
        S[f"to_r{adj}"] += add
        left -= add
        if left <= EPS:
            return S

        def use_r(e, r):
            if e <= 0:
                return 0.0
            rem = max(0.0, self.calc_vol_rem(r))
            add = min(e, rem)
            self.v_reg[r] += add
            S[f"to_r{r}"] += add
            return e - add

        def use_s12(e):
            if e <= 0:
                return 0.0
            rem = max(0.0, self.calc_vol_rem_shared_region(1, 2))
            add = min(e, rem)
            self.v_shared_12 += add
            S["to_s12"] += add
            return e - add

        def use_s23(e):
            if e <= 0:
                return 0.0
            rem = max(0.0, self.calc_vol_rem_shared_region(2, 3))
            add = min(e, rem)
            self.v_shared_23 += add
            S["to_s23"] += add
            return e - add

        def use_s123(e):
            if e <= 0:
                return 0.0
            rem = max(0.0, self.calc_rem_vol_all_shared_region())
            add = min(e, rem)
            self.v_shared_123 += add
            S["to_s123"] += add
            return e - add

        pair = {cur, adj}
        if pair == {1, 2}:
            left = use_s12(left)
            if left > EPS:
                if self.h0 > self.v3:
                    pass
                else:
                    left = use_r(left, 3)
                    if left > EPS:
                        left = use_s23(left)
                    if left > EPS:
                        left = use_s123(left)
        elif pair == {2, 3}:
            left = use_s23(left)
            if left > EPS:
                if self.h1 > self.v1:
                    pass
                else:
                    left = use_r(left, 1)
                    if left > EPS:
                        left = use_s12(left)
                    if left > EPS:
                        left = use_s123(left)
        else:
            if adj == 2:
                left = use_s12(left) if cur == 1 else use_s23(left)
                if left > EPS:
                    left = use_r(left, 1 if cur == 3 else 3)
                if left > EPS:
                    left = use_s123(left)

        if left > EPS:
            self.v_leaked += left
            S["leaked"] += left
        return S

    def inject_multi_region(self, volumes_per_region: Dict[int, float]) -> dict:
        """
        Inject CO2 into multiple regions simultaneously.
        
        Args:
            volumes_per_region: Dict mapping region (1,2,3) to volume to inject
        
        Returns:
            Combined result dict with totals
        """
        total_result = {"to_r1": 0.0, "to_r2": 0.0, "to_r3": 0.0, 
                       "to_s12": 0.0, "to_s23": 0.0, "to_s123": 0.0, "leaked": 0.0}
        
        for region, volume in volumes_per_region.items():
            if volume > 0:
                xloc = self.region_injector_locs_sim[region]
                result = self.inject(xloc, volume)
                for key in total_result:
                    total_result[key] += result[key]
        
        return total_result

    def is_valid_geometry(self, tol: float = 1e-8) -> bool:
        y = self.h
        idx = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
        idx = idx[(idx > 0) & (idx < len(y) - 1)]
        if len(idx) < 2:
            return False
        i1, i3 = int(idx[0]), int(idx[1])
        if abs(self.x[i3] - self.x[i1]) < self.dx:
            return False
        v1, v3 = float(y[i1]), float(y[i3])
        if (abs(v1 - self.h0) < tol) or (abs(v1 - self.h1) < tol):
            return False
        if (abs(v3 - self.h0) < tol) or (abs(v3 - self.h1) < tol):
            return False
        return True

    # ------------------------ Part D: Gravity Engines ------------------------
    
    def _init_gravity_grid(self, nz_cells=50, ns_sensors=60, sensor_x_range=(-0.4, 1.4)):
        """Initializes surface gravity grid."""
        self.grav_nz_cells = self.nz_cells
        self.grav_nx_cells = self.nx_cells
        
        phys_x = self.scaler.to_phys_x(self.x)
        phys_dx = phys_x[1] - phys_x[0]
        
        sim_h_max = self.hmax_all
        sim_z_grid = np.linspace(0, sim_h_max, nz_cells)
        sim_dz = sim_z_grid[1] - sim_z_grid[0]
        
        phys_dz_cell = sim_dz * self.scaler.y_scale
        self.grav_cell_volume_m3 = phys_dx * phys_dz_cell * self.scaler.z_thick

        xx_sim, hh_sim = np.meshgrid(self.x, sim_z_grid)
        flat_x_sim = xx_sim.ravel()
        flat_h_sim = hh_sim.ravel()
        
        self.grav_cell_x_m = self.scaler.to_phys_x(flat_x_sim)
        self.grav_cell_z_m = self.scaler.to_phys_depth(flat_h_sim)
        self.grav_cell_y_m = np.zeros_like(self.grav_cell_x_m)

        N_cells = len(self.grav_cell_x_m)

        sx_min, sx_max = sensor_x_range
        self.grav_sensor_x_sim = np.linspace(sx_min, sx_max, ns_sensors)
        sensor_x_m = self.scaler.to_phys_x(self.grav_sensor_x_sim)
        sensor_z_m = np.zeros(ns_sensors)
        sensor_y_m = np.zeros(ns_sensors)

        self.grav_geo_matrix = np.zeros((N_cells, ns_sensors))
        epsilon_sq = 1.0

        for j in range(ns_sensors):
            dx = self.grav_cell_x_m - sensor_x_m[j]
            dy = self.grav_cell_y_m - sensor_y_m[j]
            dz = self.grav_cell_z_m - sensor_z_m[j]
            
            r_sq = dx**2 + dy**2 + dz**2 + epsilon_sq
            r = np.sqrt(r_sq)
            r_cubed = r**3
            
            self.grav_geo_matrix[:, j] = dz / r_cubed

        topo_h_at_cells = np.interp(flat_x_sim, self.x, self.h)
        self.grav_is_reservoir_mask = (flat_h_sim < topo_h_at_cells)

    def _init_borehole_gravity(self, nz_sensors=50):
        """Initializes borehole gravity grid."""
        self.bh_sensor_z_m = np.linspace(self.scaler.top_depth, self.scaler.base_depth, nz_sensors)
        ns_bh = len(self.bh_sensor_z_m)
        
        bh_x_m = self.obs_well_loc_m
        bh_y_m = 0.0
        N_cells = len(self.grav_cell_x_m)
        
        self.borehole_geo_matrix_x = np.zeros((N_cells, ns_bh))
        self.borehole_geo_matrix_z = np.zeros((N_cells, ns_bh))
        epsilon_sq = 1.0
        
        for j in range(ns_bh):
            z_g = self.bh_sensor_z_m[j]
            x_g = bh_x_m
            
            diff_x = self.grav_cell_x_m - x_g
            diff_y = self.grav_cell_y_m - bh_y_m
            diff_z = self.grav_cell_z_m - z_g
            
            r_sq = diff_x**2 + diff_y**2 + diff_z**2 + epsilon_sq
            r = np.sqrt(r_sq)
            r_cubed = r**3
            
            self.borehole_geo_matrix_x[:, j] = diff_x / r_cubed
            self.borehole_geo_matrix_z[:, j] = diff_z / r_cubed

    def get_time_lapse_gravity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates SURFACE gravity (Vertical component only)."""
        current_base_sim = self._compute_current_base()
        cell_h_sim = (self.scaler.base_depth - self.grav_cell_z_m) / self.scaler.y_scale
        cell_x_sim = self.grav_cell_x_m / self.scaler.x_scale
        fluid_base_at_cells = np.interp(cell_x_sim, self.x, current_base_sim)
        
        is_filled = self.grav_is_reservoir_mask & (cell_h_sim > fluid_base_at_cells)
        
        mass_change = self.grav_cell_volume_m3 * self.params.rho * self.density_diff * is_filled.astype(float)
        
        delta_g_SI = self.G * (mass_change @ self.grav_geo_matrix)
        delta_g_SI = np.nan_to_num(delta_g_SI)
        return self.scaler.to_phys_x(self.grav_sensor_x_sim), delta_g_SI * 1e8

    def get_borehole_gravity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates BOREHOLE gravity."""
        current_base_sim = self._compute_current_base()
        
        cell_h_sim = (self.scaler.base_depth - self.grav_cell_z_m) / self.scaler.y_scale
        cell_x_sim = self.grav_cell_x_m / self.scaler.x_scale
        fluid_base_at_cells = np.interp(cell_x_sim, self.x, current_base_sim)
        
        is_filled = self.grav_is_reservoir_mask & (cell_h_sim > fluid_base_at_cells)
        
        mass_change = self.grav_cell_volume_m3 * self.params.rho * self.density_diff * is_filled.astype(float)
        
        delta_gx_SI = self.G * (mass_change @ self.borehole_geo_matrix_x)
        delta_gx_uGal = np.nan_to_num(delta_gx_SI) * self.si_to_ugal
        
        delta_gz_SI = self.G * (mass_change @ self.borehole_geo_matrix_z)
        delta_gz_uGal = np.nan_to_num(delta_gz_SI) * self.si_to_ugal
        
        return self.bh_sensor_z_m, delta_gx_uGal, delta_gz_uGal

    # --------------------------------------------------------------------------
    # PART 5: State computation and utilities
    # --------------------------------------------------------------------------
    
    def _level_region(self, r: int) -> float:
        mask = [None, self.mask1, self.mask2, self.mask3][r]
        thresh = [None, self.thresh1, self.thresh2, self.thresh3][r]
        hmax = [None, self.hmax1, self.hmax2, self.hmax3][r]
        return self._invert_volume_to_level(mask, thresh, hmax, self.v_reg[r])

    def _level_s12(self) -> Optional[float]:
        if self.v_shared_12 <= 0 or self.cap12 <= 0:
            return None
        mask = self.mask1 | self.mask2
        Vtop = self._volume_region_at_level(mask, self.top12)
        return self._invert_volume_to_level(mask, self.bot12, self.top12, self.v_shared_12 + Vtop)

    def _level_s23(self) -> Optional[float]:
        if self.v_shared_23 <= 0 or self.cap23 <= 0:
            return None
        mask = self.mask2 | self.mask3
        Vtop = self._volume_region_at_level(mask, self.top23)
        return self._invert_volume_to_level(mask, self.bot23, self.top23, self.v_shared_23 + Vtop)

    def _level_s123(self) -> Optional[float]:
        if self.v_shared_123 <= 0 or self.cap123 <= 0:
            return None
        mask = np.ones_like(self.x, bool)
        Vtop = self._volume_region_at_level(mask, self.top123)
        return self._invert_volume_to_level(mask, self.bot123, self.top123, self.v_shared_123 + Vtop)

    def _compute_current_base(self) -> np.ndarray:
        base = np.full_like(self.h, self.hmax_all + 1.0)
        for r, mask in [(1, self.mask1), (2, self.mask2), (3, self.mask3)]:
            d = self._level_region(r)
            base[mask] = np.minimum(base[mask], d)
        d12 = self._level_s12()
        if d12 is not None:
            base[self.mask1 | self.mask2] = np.minimum(base[self.mask1 | self.mask2], d12)
        d23 = self._level_s23()
        if d23 is not None:
            base[self.mask2 | self.mask3] = np.minimum(base[self.mask2 | self.mask3], d23)
        d123 = self._level_s123()
        if d123 is not None:
            base = np.minimum(base, d123)
        base[self.mask1] = np.maximum(base[self.mask1], self.h0)
        base[self.mask3] = np.maximum(base[self.mask3], self.h1)
        return base

    def trapped_volume(self) -> float:
        return (self.v_reg[1] + self.v_reg[2] + self.v_reg[3]
                + self.v_shared_12 + self.v_shared_23 + self.v_shared_123)

    def reset_fluid_state(self):
        """Reset the fluid state to empty (for new episode with same geology)."""
        self.v_reg = {1: 0.0, 2: 0.0, 3: 0.0}
        self.v_shared_12 = 0.0
        self.v_shared_23 = 0.0
        self.v_shared_123 = 0.0
        self.v_leaked = 0.0

    def clone(self) -> 'SpillpointEnv':
        new_env = SpillpointEnv(
            self.params,
            nx_cells=self.nx_cells,
            obs_well_loc_m=self.obs_well_loc_m,
            nz_cells=self.nz_cells,
            nx_sensors=self.nx_sensors,
            nz_sensors=self.nz_sensors
        )
        new_env.v_reg = self.v_reg.copy()
        new_env.v_shared_12 = self.v_shared_12
        new_env.v_shared_23 = self.v_shared_23
        new_env.v_shared_123 = self.v_shared_123
        new_env.v_leaked = self.v_leaked
        new_env.h = self.h.copy()
        new_env.xv1, new_env.v1, new_env.xv3, new_env.v3 = self.xv1, self.v1, self.xv3, self.v3
        return new_env


    # --------------------------------------------------------------------------
    # PART 6: Visualization
    # --------------------------------------------------------------------------

    def plot_state_with_bars_and_gravity(self, step_title: str = "",
                                          grav_ylim: Optional[Tuple[float, float]] = None,
                                          injector_locs_m: Optional[Dict[int, float]] = None):
        """
        Four-panel visualization: cross-section, CO2 bar chart, surface gravity,
        borehole gravity. Adapted from spillenv1.py for the v2 environment.
        """
        fig = plt.figure(figsize=(22, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 2, 1], wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
        axb = fig.add_subplot(gs[0, 1])
        axg = fig.add_subplot(gs[0, 2])
        axbh = fig.add_subplot(gs[0, 3])

        # 1. Cross Section
        phys_x = self.scaler.to_phys_x(self.x)
        phys_top = self.scaler.to_phys_depth(self.h)
        phys_base_res = np.full_like(phys_x, self.scaler.base_depth)
        sim_base = self._compute_current_base()
        phys_fluid_contact = self.scaler.to_phys_depth(sim_base)

        ax.plot(phys_x, phys_top, 'k', lw=2, label='Caprock')
        fill_bottom = np.minimum(phys_fluid_contact, phys_base_res)
        ax.fill_between(phys_x, phys_top, fill_bottom,
                        where=(fill_bottom > phys_top),
                        color="#4c78a8", alpha=0.8, label="CO₂ Plume")
        ax.scatter([self.scaler.to_phys_x(self.xv1),
                    self.scaler.to_phys_x(self.xv3)],
                   [self.scaler.to_phys_depth(self.v1),
                    self.scaler.to_phys_depth(self.v3)],
                   color='red', marker='x', zorder=5, s=100, label="Spill Points")
        ax.axvline(self.obs_well_loc_m, color='green', linestyle=':',
                   linewidth=2, label="Obs Well")

        if injector_locs_m:
            # colors_inj = ['blue', 'orange', 'purple']
            # for i, (region, loc_m) in enumerate(injector_locs_m.items()):
            #     ax.axvline(loc_m, color=colors_inj[i], linestyle='--',
            #               linewidth=2, label=f"Injector R{region}")
            colors_inj = ['black', 'black', 'black']
            for i, (region, loc_m) in enumerate(injector_locs_m.items()):
                ax.axvline(loc_m, color=colors_inj[i], linestyle='--',
                          linewidth=2, label=f"Injector well")

        ax.set_title(step_title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.legend(loc="lower left", fontsize='small')
        ax.grid(True, alpha=0.3)

        # 2. Bar Chart
        trapped_mm = self.scaler.to_phys_vol(self.trapped_volume()) / 1e6
        leaked_mm = self.scaler.to_phys_vol(self.v_leaked) / 1e6
        bars = axb.bar(["Trapped", "Leaked"], [trapped_mm, leaked_mm],
                      color=["#4c78a8", "#d62728"], alpha=0.85)
        axb.set_ylim(0, max((trapped_mm + leaked_mm) * 1.2, 10.0))
        axb.set_title("CO₂ Balance")
        axb.set_ylabel("Volume (Mm³)")
        axb.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, [trapped_mm, leaked_mm]):
            axb.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha='center', va='bottom', fontsize=10)

        # 3. Surface Gravity
        grav_x, grav_y = self.get_time_lapse_gravity()
        axg.plot(grav_x, grav_y, 'm-o', markersize=4)
        axg.set_title("Time-Lapse Gravity")
        axg.set_xlabel("Position (m)")
        axg.set_ylabel(r"$\Delta g$ ($\mu$Gal)")
        axg.grid(True)
        if grav_ylim:
            axg.set_ylim(grav_ylim)

        # 4. Borehole Gravity
        bh_z, bh_gx, bh_gz = self.get_borehole_gravity()
        axbh.plot(bh_gz, bh_z, 'g-o', markersize=4, label=r'$g_z$ (Vert)')
        axbh.plot(bh_gx, bh_z, 'orange', linestyle='--', marker='x',
                  markersize=4, label=r'$g_x$ (Horiz)')
        axbh.set_title("Borehole Gravity")
        axbh.set_xlabel(r"$\Delta g$ ($\mu$Gal)")
        axbh.set_ylabel("Depth (m)")
        axbh.invert_yaxis()
        axbh.grid(True)
        axbh.legend(fontsize='x-small')

        plt.tight_layout()
        return fig, (ax, axb, axg, axbh)

    def plot_cross_section(self, ax=None, title: str = "",
                           injector_locs_m: Optional[Dict[int, float]] = None,
                           show_regions: bool = True):
        """
        Plot just the cross-section with CO2 fill and region shading.
        Lighter-weight than the full 4-panel plot, suitable for GIF frames.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.get_figure()

        phys_x = self.scaler.to_phys_x(self.x)
        phys_top = self.scaler.to_phys_depth(self.h)
        phys_base_res = np.full_like(phys_x, self.scaler.base_depth)
        sim_base = self._compute_current_base()
        phys_fluid_contact = self.scaler.to_phys_depth(sim_base)

        # Shade the three regions with light background colours
        if show_regions:
            region_colors = ['#cce5ff', '#d4edda', '#fff3cd']  # light blue, green, yellow
            region_labels = ['Region 1', 'Region 2', 'Region 3']
            for mask, col, lbl in zip(
                    [self.mask1, self.mask2, self.mask3],
                    region_colors, region_labels):
                ax.fill_between(phys_x, phys_top, phys_base_res,
                                where=mask, color=col, alpha=0.35, label=lbl)

        # CO2 plume
        fill_bottom = np.minimum(phys_fluid_contact, phys_base_res)
        ax.fill_between(phys_x, phys_top, fill_bottom,
                        where=(fill_bottom > phys_top),
                        color="#4c78a8", alpha=0.85, label="CO₂ Plume")

        # Caprock line on top
        ax.plot(phys_x, phys_top, 'k', lw=2, label='Caprock')

        # Spillpoints
        ax.scatter([self.scaler.to_phys_x(self.xv1),
                    self.scaler.to_phys_x(self.xv3)],
                   [self.scaler.to_phys_depth(self.v1),
                    self.scaler.to_phys_depth(self.v3)],
                   color='red', marker='x', zorder=5, s=120, linewidths=2.5,
                   label="Spill Points")

        # Region boundary lines
        for xsim in [self.xv1, self.xv3]:
            ax.axvline(self.scaler.to_phys_x(xsim), color='grey',
                       linestyle=':', linewidth=1, alpha=0.6)

        # Injectors
        if injector_locs_m:
            # inj_colors = ['#0d6efd', '#fd7e14', '#6f42c1']
            inj_colors = ['black', 'black', 'black']
            for i, (region, loc_m) in enumerate(injector_locs_m.items()):
                # ax.axvline(loc_m, color=inj_colors[i], linestyle='--',
                #           linewidth=1.8, label=f"Injector R{region}")
                ax.axvline(loc_m, color=inj_colors[i], linestyle='--',
                          linewidth=1.8, label="Injector well")

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.legend(loc="lower left", fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)

        return fig, ax


def build_valid_env(start_seed: int,
                    max_tries: int = 500,
                    obs_well_loc: float = 11000.0,
                    nx_cells: int = 100,
                    nz_cells: int = 100,
                    nx_sensors: int = 60,
                    nz_sensors: int = 50) -> SpillpointEnv:
    """Build a SpillpointEnv with valid geometry."""
    seed = start_seed
    for _ in range(max_tries):
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        params = sample_prior(1, rng=rng)[0]
        
        env = SpillpointEnv(params, obs_well_loc_m=obs_well_loc,
                           nx_cells=nx_cells, nz_cells=nz_cells,
                           nx_sensors=nx_sensors, nz_sensors=nz_sensors)
        
        if env.is_valid_geometry():
            print(f"Found valid geometry with seed: {seed}")
            return env
        seed += 1
    raise RuntimeError("Failed to find valid geometry.")


print("SpillpointEnv code complete!")


# ==============================================================================
# 4. GYMNASIUM ENVIRONMENT - MULTI-REGION INJECTION (v2)
# ==============================================================================

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("Warning: Neither gymnasium nor gym is available. Using fallback base class.")
        
        class FallbackEnv:
            metadata = {}
            observation_space = None
            action_space = None
            
            def reset(self, seed=None, options=None):
                pass
            
            def step(self, action):
                pass
            
            def render(self):
                pass
            
            def close(self):
                pass
        
        class spaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=np.float32):
                    self.low = np.array(low) if not isinstance(low, np.ndarray) else low
                    self.high = np.array(high) if not isinstance(high, np.ndarray) else high
                    self.shape = shape if shape else self.low.shape
                    self.dtype = dtype
                
                def sample(self):
                    return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
            
            class Discrete:
                def __init__(self, n):
                    self.n = n
                
                def sample(self):
                    return np.random.randint(0, self.n)
        
        gym = type('gym', (), {'Env': FallbackEnv})()

if GYM_AVAILABLE:
    BaseEnvClass = gym.Env
else:
    BaseEnvClass = FallbackEnv


class MultiRegionCO2StorageEnv(BaseEnvClass):
    """
    Multi-Region CO2 Storage Environment (v2)
    
    Key Features:
    1. FIXED SEED - Same geology for ALL episodes (seed doesn't change)
    2. 3 injector locations (one per region)
    3. DISCRETE ACTION SPACE - Agent selects from predefined injection rate tuples
    4. CONFIGURABLE OBSERVATIONS - Choose timelapse gravity, borehole gravity, or both
    5. REWARD: (trapped CO2 - leaked CO2) in Mm³ per step
    6. Leakage does NOT terminate the episode; max_steps=150 truncation only
    
    Action Space: Discrete(N_DISCRETE_ACTIONS)
        - Each action corresponds to a predefined (r1, r2, r3) injection rate tuple
        
    Observation Space: Depends on obs_mode parameter
        - 'timelapse': Only time-lapse gravity measurements (nx_sensors,)
        - 'borehole': Only borehole gravity measurements (nz_sensors * 2,) [gz and gx]
        - 'both': Both measurements concatenated (nx_sensors + nz_sensors * 2,)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 fixed_seed: int = 42,
                 obs_mode: str = 'both',  # 'timelapse', 'borehole', or 'both'
                 obs_well_loc_m: float = 10000.0,
                 nx_cells: int = 100,
                 nz_cells: int = 100,
                 nx_sensors: int = 30,
                 nz_sensors: int = 30,
                 max_steps: int = 20000,
                 render_mode: Optional[str] = None):
        
        super(MultiRegionCO2StorageEnv, self).__init__()

        # --- FIXED seed - same geology for ALL episodes ---
        self.fixed_seed = fixed_seed
        
        # --- Observation mode ---
        assert obs_mode in ['timelapse', 'borehole', 'both'], \
            f"obs_mode must be 'timelapse', 'borehole', or 'both', got '{obs_mode}'"
        self.obs_mode = obs_mode
        
        self.obs_well_loc = obs_well_loc_m
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Simulation config
        self.sim_config = {
            'nx_cells': nx_cells,
            'nz_cells': nz_cells,
            'nx_sensors': nx_sensors,
            'nz_sensors': nz_sensors
        }
        self.nx_sensors = nx_sensors
        self.nz_sensors = nz_sensors

        # --- DISCRETE Action Space ---
        # Agent selects an index into DISCRETE_ACTIONS list
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        # --- Observation Space (POMDP - only gravity measurements) ---
        if obs_mode == 'timelapse':
            obs_size = nx_sensors
        elif obs_mode == 'borehole':
            obs_size = nz_sensors * 2  # gz and gx components
        else:  # 'both'
            obs_size = nx_sensors + nz_sensors * 2
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        self.obs_size = obs_size

        # Build simulation environment ONCE with fixed seed
        print(f"Building environment with FIXED seed: {self.fixed_seed}")
        print(f"Observation mode: {self.obs_mode} (obs_size={obs_size})")
        self.sim_env = build_valid_env(
            start_seed=self.fixed_seed,
            obs_well_loc=self.obs_well_loc,
            **self.sim_config
        )
        
        # Store injector locations
        self.injector_locs_m = self.sim_env.get_region_injector_locations_m()
        print(f"Injector locations: R1={self.injector_locs_m[1]:.0f}m, "
              f"R2={self.injector_locs_m[2]:.0f}m, R3={self.injector_locs_m[3]:.0f}m")
        
        # Internal state
        self.current_step = 0
        self.total_trapped_m3 = 0.0
        self.total_leaked_m3 = 0.0
        self.region_trapped_m3 = {1: 0.0, 2: 0.0, 3: 0.0}
        
        # Cache for gravity (computed once per step)
        self._cached_gravity = None
        self._gravity_step = -1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment for a new episode.
        
        NOTE: The geology stays the SAME (fixed seed). Only fluid state resets.
        """
        super().reset(seed=seed)
        
        # Reset fluid state (but keep same geology!)
        self.sim_env.reset_fluid_state()
        
        # Reset counters
        self.current_step = 0
        self.total_trapped_m3 = 0.0
        self.total_leaked_m3 = 0.0
        self.region_trapped_m3 = {1: 0.0, 2: 0.0, 3: 0.0}
        
        # Clear gravity cache
        self._cached_gravity = None
        self._gravity_step = -1
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "fixed_seed": self.fixed_seed,
            "obs_mode": self.obs_mode,
            "injector_locations_m": self.injector_locs_m,
            "trapped_total_m3": 0.0,
            "leaked_total_m3": 0.0
        }
        
        return observation, info

    def step(self, action: int):
        """
        Execute one time step using discrete action.
        
        Args:
            action: int - Index into DISCRETE_ACTIONS list
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Get injection volumes from action
        action_tuple = DISCRETE_ACTIONS[action]  # (r1_vol, r2_vol, r3_vol)
        
        # 2. Convert to simulation units and inject
        scaler = self.sim_env.scaler
        volumes_sim = {}
        for region in [1, 2, 3]:
            vol_m3 = action_tuple[region - 1]  # Already in m³
            volumes_sim[region] = scaler.to_sim_vol(vol_m3)
        
        # 3. Inject CO2 into all regions
        result_dict = self.sim_env.inject_multi_region(volumes_sim)
        
        # 4. Calculate trapped and leaked volumes
        trapped_sim = sum(v for k, v in result_dict.items() if k != 'leaked')
        leaked_sim = result_dict.get('leaked', 0.0)
        
        trapped_m3 = scaler.to_phys_vol(trapped_sim)
        leaked_m3 = scaler.to_phys_vol(leaked_sim)
        
        # Track per-region trapped
        for region in [1, 2, 3]:
            region_trapped = scaler.to_phys_vol(result_dict.get(f'to_r{region}', 0.0))
            self.region_trapped_m3[region] += region_trapped
        
        self.total_trapped_m3 += trapped_m3
        self.total_leaked_m3 += leaked_m3
        
        # 5. Check for leakage (no longer terminates episode)
        has_leaked = leaked_m3 > 1e-3
        
        # 6. OLD SIMPLIFIED REWARD: +1 for no leakage, 0 for leakage
        # if has_leaked:
        #     step_reward = 0.0
        #     terminated = True
        # else:
        #     step_reward = 1.0
        #     terminated = False
        
        # 6. NEW REWARD: (trapped CO2 - leaked CO2) in Mm³
        #    Leakage no longer terminates the episode — agent must learn to avoid it.
        step_reward = (trapped_m3 - leaked_m3) / 1e6
        terminated = False
        
        # 7. Check truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # 8. Invalidate gravity cache (state changed)
        self._gravity_step = -1
        
        # 9. Get observation
        observation = self._get_observation()
        
        # 10. Build info dict
        info = {
            "step": self.current_step,
            "action_idx": action,
            "action_tuple": action_tuple,
            "trapped_step_m3": trapped_m3,
            "leaked_step_m3": leaked_m3,
            "trapped_total_m3": self.total_trapped_m3,
            "leaked_total_m3": self.total_leaked_m3,
            "has_leaked": has_leaked,
            "region_trapped_m3": self.region_trapped_m3.copy()
        }
        
        return observation, float(step_reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector with gravity measurements only (POMDP)."""
        # Use cached gravity if available for current step
        if self._gravity_step != self.current_step:
            # Compute and cache gravity measurements
            _, surf_grav = self.sim_env.get_time_lapse_gravity()
            _, bh_gx, bh_gz = self.sim_env.get_borehole_gravity()
            self._cached_gravity = (surf_grav, bh_gz, bh_gx)
            self._gravity_step = self.current_step
        
        surf_grav, bh_gz, bh_gx = self._cached_gravity
        
        # Build observation based on mode
        if self.obs_mode == 'timelapse':
            obs = surf_grav
        elif self.obs_mode == 'borehole':
            obs = np.concatenate([bh_gz, bh_gx])
        else:  # 'both'
            obs = np.concatenate([surf_grav, bh_gz, bh_gx])
        
        return obs.astype(np.float32)

    def render(self):
        """Render current state."""
        if self.sim_env is None:
            return None
            
        # title = (f"Step {self.current_step} | "
        #         f"Trapped: {self.total_trapped_m3/1e6:.2f} Mm³ | "
        #         f"Leaked: {self.total_leaked_m3/1e6:.4f} Mm³")
        title = (f"Trapped: {self.total_trapped_m3/1e6:.2f} Mm³ | "
                f"Leaked: {self.total_leaked_m3/1e6:.4f} Mm³")
        
        fig, _ = self.sim_env.plot_state_with_bars_and_gravity(
            title,
            injector_locs_m=self.injector_locs_m
        )
        
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data
        else:
            return fig

    # ------------------------------------------------------------------
    # Visualization: replay a list of actions as static plots or GIF
    # ------------------------------------------------------------------

    def _make_frame(self, step: int, action_idx: Optional[int],
                    grav_ylim: Optional[Tuple[float, float]] = None,
                    show_wells: Optional[List[int]] = None,
                    figsize: Tuple[float, float] = (18, 8)):
        """
        Produce a single figure for the current environment state.
        
        Layout (2 rows, equal height):
            Top row:  spillpoint cross-section (depth 0 to base_depth)
            Bottom row: timelapse gravity (full width)
        
        Args:
            step: current step number (0 = initial empty state)
            action_idx: the discrete action taken to reach this state (None for step 0)
            grav_ylim: optional fixed y-limits for gravity panel (keeps GIF stable)
            show_wells: list of region numbers whose injector wells to show, e.g. [1,2,3].
                        Default [1,2,3]. Pass [] to hide all wells.
            figsize: figure size
            
        Returns:
            matplotlib Figure
        """
        if show_wells is None:
            show_wells = [1, 2, 3]

        sim = self.sim_env
        scaler = sim.scaler

        fig = plt.figure(figsize=figsize, facecolor='white')
        gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.45)

        # ---- Top: cross-section (full width) ----
        ax_cs = fig.add_subplot(gs_main[0])
        ax_cs.set_facecolor('white')

        phys_x = scaler.to_phys_x(sim.x)
        phys_top = scaler.to_phys_depth(sim.h)
        phys_base_res = np.full_like(phys_x, scaler.base_depth)
        sim_base = sim._compute_current_base()
        phys_fluid = scaler.to_phys_depth(sim_base)
        # phys_plot_base = np.full_like(phys_x, 900.0)   # new
        phys_plot_base = np.full_like(phys_x, 2600.0)   # new

        # # Original code start
        # # # Brown overburden rock: fills between caprock (top surface) and
        # # # depth 0 (ground surface), only within x=[0, 20000].
        # # # This represents rock layers above the caprock but below the surface.
        # # ax_cs.fill_between(phys_x, 0, phys_top,
        # #                    color='#8B6914', alpha=0.70, label='Overburden')
        # # Original code end

        # # # New code start
        # # # Extended Overburden
        # # ext_x = np.concatenate([[-6000], phys_x, [scaler.x_scale + 6000]])
        # # ext_top = np.concatenate([[float(phys_top[0])], phys_top, [float(phys_top[-1])]])
        # # ax_cs.fill_between(ext_x, 0, ext_top,
        # #                    color='#8B6914', alpha=0.70, label='Overburden')
        # # # New code end

        # # ---------- Extended Boundary Arrays ----------
        # # Stretch the arrays out to the padded visual edges
        # ext_x = np.concatenate([[-6000], phys_x, [scaler.x_scale + 6000]])
        # ext_top = np.concatenate([[float(phys_top[0])], phys_top, [float(phys_top[-1])]])
        # ext_plot_base = np.full_like(ext_x, 900.0)

        # # 1. Extended Overburden
        # ax_cs.fill_between(ext_x, 0, ext_top,
        #                    color='#8B6914', alpha=0.70, label='Overburden')

        # # 2. Extended Brine
        # ax_cs.fill_between(ext_x, ext_top, ext_plot_base,
        #                    where=(ext_plot_base > ext_top),
        #                    color='#ADD8E6', alpha=0.60, label='Brine')

        # # Uniform light-blue brine fill (replaces per-region colors)
        # # ax_cs.fill_between(phys_x, phys_top, phys_base_res,
        # #                    color='#ADD8E6', alpha=0.60, label='Brine')
        # # # Original start
        # # ax_cs.fill_between(phys_x, phys_top, phys_base_res,
        # #                    where=(phys_base_res > phys_top),
        # #                    color='#ADD8E6', alpha=0.60, label='Brine')
        # # # Original end

        # # New code start
        # # Uniform light-blue brine fill down to 900m
        # ax_cs.fill_between(phys_x, phys_top, phys_plot_base,
        #                    where=(phys_plot_base > phys_top),
        #                    color='#ADD8E6', alpha=0.60, label='Brine')
        # # New code end

        # # CO2 plume (orange)
        # fill_bottom = np.minimum(phys_fluid, phys_base_res)
        # ax_cs.fill_between(phys_x, phys_top, fill_bottom,
        #                    where=(fill_bottom > phys_top),
        #                    color='#FF8C00', alpha=0.90, label="CO₂ Plume")

        # # # Caprock — gray, thick (anticline line only, no base line)
        # # ax_cs.plot(phys_x, phys_top, color='gray', lw=5, label='Caprock',
        # #            zorder=4, solid_capstyle='round')

        # # # new code start
        # # # Caprock (thickened upwards by 20 meters)
        # # caprock_thickness = 100.0
        # # phys_top_upper = phys_top - caprock_thickness
        
        # # ax_cs.fill_between(phys_x, phys_top_upper, phys_top, 
        # #                    color='gray', zorder=4, label='Caprock')
        # caprock_thickness = 100.0
        # ext_top_upper = ext_top - caprock_thickness
        # ax_cs.fill_between(ext_x, ext_top_upper, ext_top, 
        #                    color='gray', zorder=4, label='Caprock')
        # # # new code end

        # caprock_thickness = 100.0
        # ext_top_upper = ext_top - caprock_thickness
        # ax_cs.fill_between(ext_x, ext_top_upper, ext_top, 
        #                    color='gray', zorder=4, label='Caprock')


        # ---------- Extended Boundary Arrays ----------
        ext_x = np.concatenate([[-6000], phys_x, [scaler.x_scale + 6000]])
        ext_top = np.concatenate([[float(phys_top[0])], phys_top, [float(phys_top[-1])]])
        # ext_plot_base = np.full_like(ext_x, 900.0)
        ext_plot_base = np.full_like(ext_x, 2600.0)

        # 1. Extended Overburden
        ax_cs.fill_between(ext_x, 0, ext_top,
                           color='#8B6914', alpha=0.70, label='Overburden')

        # 2. Extended Brine
        ax_cs.fill_between(ext_x, ext_top, ext_plot_base,
                           where=(ext_plot_base > ext_top),
                           color='#ADD8E6', alpha=0.60, label='Brine')

        # 3. CO2 Plume (Stays restricted to the central physics area)
        fill_bottom = np.minimum(phys_fluid, phys_base_res)
        ax_cs.fill_between(phys_x, phys_top, fill_bottom,
                           where=(fill_bottom > phys_top),
                           color='#FF8C00', alpha=0.90, label="CO₂ Plume")
        
        # 4. Extended Caprock
        caprock_thickness = 100.0
        ext_top_upper = ext_top - caprock_thickness
        ax_cs.fill_between(ext_x, ext_top_upper, ext_top, 
                           color='gray', zorder=4, label='Caprock')

        # ---------- Faults at x=0 and x=20000 ----------
        # # We add zorder=5 here to force the lines on top of the Overburden
        # ax_cs.plot([0, 0], [0, 2600], color='black', lw=2.5, solid_capstyle='round', zorder=5)
        # ax_cs.plot([scaler.x_scale, scaler.x_scale], [0, 2600], color='black', lw=2.5, solid_capstyle='round', zorder=5)
        
        
        # ---------- Tilted Faults with Half-Arrows ----------
        # We add zorder=5 here to force the lines on top of the Overburden
        def add_half_arrow(ax, x_start, z_start, x_end, z_end, barb_side):
            """Helper to draw a structural geology half-arrow"""
            # Draw the main shaft
            ax.plot([x_start, x_end], [z_start, z_end], color='black', lw=1.5, zorder=5)
            
            # Calculate the vector pointing backward from the tip
            dx = x_start - x_end
            dz = z_start - z_end
            L = np.hypot(dx, dz)
            if L == 0: return
            ux, uz = dx/L, dz/L
            
            # Calculate the perpendicular vector to place the barb on one side
            px, pz = -uz, ux
            if barb_side == 'right':
                px, pz = uz, -ux
                
            # Combine to get a barb angled ~30 degrees away from the shaft
            bx = 0.866 * ux + 0.5 * px
            bz = 0.866 * uz + 0.5 * pz
            
            barb_len = 250  # length of barb in meters
            ax.plot([x_end, x_end + barb_len*bx], [z_end, z_end + barb_len*bz], color='black', lw=1.5, zorder=5)

        # 1. Left Fault
        # left_fault_depth = float(phys_top[0]) # Dynamically grab exact caprock depth at x=0
        left_fault_depth = float(2600)
        left_top_x = -1000
        left_bot_x = 0
        ax_cs.plot([left_top_x, left_bot_x], [0, left_fault_depth], color='black', lw=2.5, solid_capstyle='round', zorder=5)
        
        # Left Fault Slip Arrows
        mx, mz = (left_top_x + left_bot_x) / 2, left_fault_depth / 2
        vx, vz = left_bot_x - left_top_x, left_fault_depth - 0
        L = np.hypot(vx, vz)
        ux, uz = vx/L, vz/L
        
        O = 350  # Offset distance from the fault line in meters
        A = 300  # Half-length of the arrow shaft
        ox, oz = O * (-uz), O * (ux)    # Outside offset vector
        ix, iz = O * (uz), O * (-ux)    # Inside offset vector
        
        # Outside half-arrow (footwall - moving UP)
        add_half_arrow(ax_cs, mx + ox + A*ux, mz + oz + A*uz, mx + ox - A*ux, mz + oz - A*uz, 'left')
        # Inside half-arrow (hanging wall - moving DOWN)
        add_half_arrow(ax_cs, mx + ix - A*ux, mz + iz - A*uz, mx + ix + A*ux, mz + iz + A*uz, 'right')


        # 2. Right Fault
        # right_fault_depth = float(phys_top[-1]) # Dynamically grab exact caprock depth at x=20000
        right_fault_depth = float(2600)
        right_bot_x = scaler.x_scale # This is exactly 20000
        right_top_x = scaler.x_scale + 1000
        ax_cs.plot([right_top_x, right_bot_x], [0, right_fault_depth], color='black', lw=2.5, solid_capstyle='round', zorder=5)
        
        # Right Fault Slip Arrows
        mx, mz = (right_top_x + right_bot_x) / 2, right_fault_depth / 2
        vx, vz = right_bot_x - right_top_x, right_fault_depth - 0
        L = np.hypot(vx, vz)
        ux, uz = vx/L, vz/L
        
        ox, oz = O * (uz), O * (-ux)    # Outside offset vector
        ix, iz = O * (-uz), O * (ux)    # Inside offset vector
        
        # Outside half-arrow (footwall - moving UP)
        add_half_arrow(ax_cs, mx + ox + A*ux, mz + oz + A*uz, mx + ox - A*ux, mz + oz - A*uz, 'right')
        # Inside half-arrow (hanging wall - moving DOWN)
        add_half_arrow(ax_cs, mx + ix - A*ux, mz + iz - A*uz, mx + ix + A*ux, mz + iz + A*uz, 'left')


        # Spillpoints
        ax_cs.scatter(
            [scaler.to_phys_x(sim.xv1), scaler.to_phys_x(sim.xv3)],
            [scaler.to_phys_depth(sim.v1), scaler.to_phys_depth(sim.v3)],
            color='red', marker='x', zorder=5, s=120, linewidths=2.5,
            label="Spill Points")

        # Region boundary lines
        for xsim in [sim.xv1, sim.xv3]:
            ax_cs.axvline(scaler.to_phys_x(xsim), color='grey',
                          linestyle=':', linewidth=1, alpha=0.6)

        # # ---------- Faults at x=0 and x=20000 ----------
        # # Each fault starts at the actual reservoir boundary depth at its
        # # respective edge, and extends outward at 45° going upward by 50m.
        # # fault_dy = 50.0   # vertical span (going upward)
        # fault_dy = 250.0   # vertical span (going upward)
        # fault_dx = 250.0   # horizontal span (45°)
        # # Left fault: starts at depth of caprock at x=0
        # left_fault_depth = float(phys_top[0])   # actual depth at x=0
        # ax_cs.plot([0, -fault_dx], [left_fault_depth, left_fault_depth - fault_dy],
        #            color='black', lw=2.5, solid_capstyle='round')
        # for frac in [0.3, 0.6]:
        #     fx = -fault_dx * frac
        #     fy = left_fault_depth - fault_dy * frac
        #     ax_cs.plot([fx, fx - 15], [fy, fy + 10],
        #                color='black', lw=1.5, solid_capstyle='round')
        # # Right fault: starts at depth of caprock at x=20000
        # x_right = scaler.x_scale
        # right_fault_depth = float(phys_top[-1])  # actual depth at x=20000
        # ax_cs.plot([x_right, x_right + fault_dx],
        #            [right_fault_depth, right_fault_depth - fault_dy],
        #            color='black', lw=2.5, solid_capstyle='round')
        # for frac in [0.3, 0.6]:
        #     fx = x_right + fault_dx * frac
        #     fy = right_fault_depth - fault_dy * frac
        #     ax_cs.plot([fx, fx + 15], [fy, fy + 10],
        #                color='black', lw=1.5, solid_capstyle='round')

        # # ---------- Faults at x=0 and x=20000 ----------
        # # Left fault
        # ax_cs.plot([0, 0], [0, 900], color='black', lw=2.5, solid_capstyle='round')
        # # Right fault
        # ax_cs.plot([scaler.x_scale, scaler.x_scale], [0, 900], color='black', lw=2.5, solid_capstyle='round')

        # ---------- Surface sensor line at depth 0 ----------
        sensor_x_phys = scaler.to_phys_x(sim.grav_sensor_x_sim)
        ax_cs.axhline(0, color='gray', lw=1.0, alpha=0.5)
        ax_cs.scatter(sensor_x_phys,
                      np.zeros(len(sensor_x_phys)),
                      color='green', marker='+', s=80, linewidths=1.8,
                      zorder=6, label='Gravity Sensors')

        # # Injector well lines (only for wells in show_wells)
        # inj_colors = {1: '#0d6efd', 2: '#fd7e14', 3: '#6f42c1'}
        # for region in show_wells:
        #     if region in self.injector_locs_m:
        #         loc_m = self.injector_locs_m[region]
        #         ax_cs.axvline(loc_m, color=inj_colors[region], linestyle='--',
        #                      linewidth=1.8, label=f"Injector R{region}")
                
        inj_colors = {1: '#000000', 2: '#000000', 3: '#000000'}
        added_label = False
        for region in show_wells:
            if region in self.injector_locs_m:
                loc_m = self.injector_locs_m[region]
                current_label = "Injector well" if not added_label else None
                ax_cs.axvline(loc_m, color=inj_colors[region], linestyle='--',
                             linewidth=1.8, label=current_label)
                added_label = True

        # Build title
        trapped_mm3 = self.total_trapped_m3 / 1e6
        leaked_mm3 = self.total_leaked_m3 / 1e6
        total_cap_mm3 = scaler.to_phys_vol(
            sim.cap1 + sim.cap2 + sim.cap3 + sim.cap12 + sim.cap23 + sim.cap123) / 1e6
        if step == 0:
            title = (f"Step 0 — Empty Reservoir  |  "
                     f"Total Capacity: {total_cap_mm3:.2f} Mm³")
        else:
            act_tuple = DISCRETE_ACTIONS[action_idx]
            total_rate = sum(act_tuple)
            fracs = tuple(v / total_rate for v in act_tuple)
            # title = (f"Step {step}  |  Action {action_idx}: "
            #          f"R1={fracs[0]:.0%} R2={fracs[1]:.0%} R3={fracs[2]:.0%}  |  "
            #          f"Trapped: {trapped_mm3:.2f} / {total_cap_mm3:.2f} Mm³  |  "
            #          f"Leaked: {leaked_mm3:.4f} Mm³")
            
            # New code
            # Calculate the fill percentage for each region
            r1_fill = (sim.v_reg[1] / sim.cap1) if sim.cap1 > 0 else 0
            r2_fill = (sim.v_reg[2] / sim.cap2) if sim.cap2 > 0 else 0
            r3_fill = (sim.v_reg[3] / sim.cap3) if sim.cap3 > 0 else 0
            
            title = (f"Step {step}  |  Filled: "
                     f"R1={r1_fill:.0%} R2={r2_fill:.0%} R3={r3_fill:.0%}  |  "
                     f"Trapped: {trapped_mm3:.2f} / {total_cap_mm3:.2f} Mm³  |  "
                     f"Leaked: {leaked_mm3:.4f} Mm³")

        ax_cs.set_title(title, fontsize=12, fontweight='bold')
        ax_cs.set_xlabel("Distance (m)")
        ax_cs.set_ylabel("Depth (m)")
        # y-axis: surface at top, base at bottom; padding above for sensors,
        # below for faults
        # ax_cs.set_ylim(scaler.base_depth + 30, -60)  # original
        # ax_cs.set_ylim(900 + 30, -60)  # new
        ax_cs.set_ylim(2600 + 30, -60)  # new
        ax_cs.set_xlim(-6000, scaler.x_scale + 6000)
        # ax_cs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
        #              fontsize='small', ncol=4, framealpha=0.9)
        # ax_cs.legend(loc="center left", fontsize='small', ncol=2, framealpha=0.9)
        # ax_cs.grid(True, alpha=0.15)

        # ---- Bottom: timelapse gravity (full width) ----
        ax_grav = fig.add_subplot(gs_main[1])
        ax_grav.set_facecolor('white')

        grav_x, grav_y = sim.get_time_lapse_gravity()
        ax_grav.plot(grav_x, grav_y, 'm-o', markersize=3, linewidth=1.5)
        # Mark region boundaries on gravity plot
        for xsim in [sim.xv1, sim.xv3]:
            ax_grav.axvline(scaler.to_phys_x(xsim), color='grey',
                            linestyle=':', linewidth=0.8, alpha=0.5)
        ax_grav.set_title("Time-Lapse Gravity")
        ax_grav.set_xlabel("Position (m)")
        ax_grav.set_ylabel(r"$\Delta g$ ($\mu$Gal)")
        ax_grav.grid(True, alpha=0.3)
        # Match x-limits with cross-section for direct comparison
        ax_grav.set_xlim(-6000, scaler.x_scale + 6000)
        if grav_ylim:
            ax_grav.set_ylim(grav_ylim)

        # Extract the handles and labels from the cross-section
        handles, labels = ax_cs.get_legend_handles_labels()
        
        # Draw them on the gravity plot instead
        ax_grav.legend(handles, labels, loc="center left", 
                       fontsize='small', ncol=2, framealpha=0.9)

        return fig

    def visualize_actions(self, actions: List[int],
                          grav_ylim: Optional[Tuple[float, float]] = None,
                          show_wells: Optional[List[int]] = None,
                          figsize: Tuple[float, float] = (18, 8)):
        """
        Replay a list of discrete actions and display static inline plots
        (designed for Jupyter notebooks — calls plt.show() for each frame).

        Shows an initial empty-reservoir frame (step 0) followed by one frame
        per action.  Stops early if leakage occurs.

        Args:
            actions: list of discrete action indices to replay
            grav_ylim: optional fixed (ymin, ymax) for gravity panels.
                       If None, automatically pre-computed from a dry run.
            show_wells: list of region numbers whose injector wells to show,
                        e.g. [1,2,3] (default). Pass [] to hide all.
            figsize: per-frame figure size

        Returns:
            list of info dicts (one per executed step)
        """
        if show_wells is None:
            show_wells = [1, 2, 3]

        # --- Pre-compute static gravity y-limits if not provided ---
        if grav_ylim is None:
            self.reset()
            grav_min, grav_max = 0.0, 0.0
            for action_idx in actions:
                obs, reward, terminated, truncated, info = self.step(action_idx)
                _, grav_y = self.sim_env.get_time_lapse_gravity()
                grav_min = min(grav_min, float(grav_y.min()))
                grav_max = max(grav_max, float(grav_y.max()))
                if terminated or truncated:
                    break
            pad = max(abs(grav_max - grav_min) * 0.10, 0.5)
            grav_ylim = (grav_min - pad, grav_max + pad)

        # Reset to empty state
        self.reset()

        # Step 0: empty reservoir
        fig = self._make_frame(step=0, action_idx=None,
                               grav_ylim=grav_ylim, show_wells=show_wells,
                               figsize=figsize)
        plt.show()

        infos = []
        for i, action_idx in enumerate(actions):
            obs, reward, terminated, truncated, info = self.step(action_idx)
            infos.append(info)

            fig = self._make_frame(step=i + 1, action_idx=action_idx,
                                   grav_ylim=grav_ylim, show_wells=show_wells,
                                   figsize=figsize)
            plt.show()

            if terminated:
                print(f"*** LEAKAGE at step {i + 1} — episode terminated ***")
                break
            if truncated:
                print(f"*** MAX STEPS reached at step {i + 1} ***")
                break

        return infos

    def create_gif(self, actions: List[int],
                   save_path: str = "co2_filling.gif",
                   fps: int = 2,
                   grav_ylim: Optional[Tuple[float, float]] = None,
                   show_wells: Optional[List[int]] = None,
                   figsize: Tuple[float, float] = (18, 8),
                   dpi: int = 100):
        """
        Replay a list of discrete actions and save an animated GIF.

        The GIF starts with the empty reservoir and adds one frame per
        action, stopping early on leakage.

        Args:
            actions: list of discrete action indices
            save_path: output file path for the GIF
            fps: frames per second
            grav_ylim: optional fixed y-limits for gravity panels.
                       If None, automatically pre-computed from a dry run.
            show_wells: list of region numbers whose injector wells to show,
                        e.g. [1,2,3] (default). Pass [] to hide all.
            figsize: figure size per frame
            dpi: resolution

        Returns:
            list of info dicts for each executed step
        """
        import io
        from PIL import Image

        if show_wells is None:
            show_wells = [1, 2, 3]

        # --- Pre-compute static gravity y-limits if not provided ---
        if grav_ylim is None:
            self.reset()
            grav_min, grav_max = 0.0, 0.0
            for action_idx in actions:
                obs, reward, terminated, truncated, info = self.step(action_idx)
                _, grav_y = self.sim_env.get_time_lapse_gravity()
                grav_min = min(grav_min, float(grav_y.min()))
                grav_max = max(grav_max, float(grav_y.max()))
                if terminated or truncated:
                    break
            pad = max(abs(grav_max - grav_min) * 0.10, 0.5)
            grav_ylim = (grav_min - pad, grav_max + pad)

        frames: List[Image.Image] = []
        infos = []

        # Reset
        self.reset()

        # Frame 0: empty
        fig = self._make_frame(step=0, action_idx=None,
                               grav_ylim=grav_ylim, show_wells=show_wells,
                               figsize=figsize)
        frames.append(self._fig_to_pil(fig, dpi))
        plt.close(fig)

        # One frame per action
        for i, action_idx in enumerate(actions):
            obs, reward, terminated, truncated, info = self.step(action_idx)
            infos.append(info)

            fig = self._make_frame(step=i + 1, action_idx=action_idx,
                                   grav_ylim=grav_ylim, show_wells=show_wells,
                                   figsize=figsize)
            frames.append(self._fig_to_pil(fig, dpi))
            plt.close(fig)

            if terminated:
                print(f"*** LEAKAGE at step {i + 1} — GIF ends here ***")
                break
            if truncated:
                print(f"*** MAX STEPS reached at step {i + 1} ***")
                break

        # Save GIF
        if frames:
            duration_ms = int(1000 / fps)
            # Hold the last frame longer so the viewer can see the final state
            durations = [duration_ms] * len(frames)
            durations[-1] = duration_ms * 4

            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
            )
            print(f"GIF saved to: {save_path}  "
                  f"({len(frames)} frames, {fps} fps)")

        return infos

    @staticmethod
    def _fig_to_pil(fig, dpi: int = 100):
        """Convert a matplotlib Figure to a PIL Image (for GIF creation)."""
        from PIL import Image
        import io
        fig.set_dpi(dpi)
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def get_action_meanings(self) -> List[str]:
        """Return human-readable descriptions of each action."""
        meanings = []
        for i, (r1, r2, r3) in enumerate(DISCRETE_ACTIONS):
            total = r1 + r2 + r3
            f1, f2, f3 = r1/total, r2/total, r3/total
            meanings.append(f"Action {i}: R1={f1:.1%}, R2={f2:.1%}, R3={f3:.1%}")
        return meanings


# Backwards compatibility alias
SpillpointPOMDPEnv = MultiRegionCO2StorageEnv


print("Multi-Region Environment (v2) code complete!")




# ==============================================================================
# ================================= NEW CODE ===================================
# ==============================================================================
def save_static_frames(env, actions, save_dir="saved_frames", dpi=150, show_wells=None):
    """
    Runs the environment through a list of actions and saves each step as a static PNG.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Pre-compute gravity y-limits to keep the graphs stable (optional but recommended)
    env.reset()
    grav_min, grav_max = 0.0, 0.0
    for action_idx in actions:
        env.step(action_idx)
        _, grav_y = env.sim_env.get_time_lapse_gravity()
        grav_min = min(grav_min, float(grav_y.min()))
        grav_max = max(grav_max, float(grav_y.max()))
    pad = max(abs(grav_max - grav_min) * 0.10, 0.5)
    grav_ylim = (grav_min - pad, grav_max + pad)

    # Reset environment to start the actual run
    env.reset()

    # Save Step 0 (Empty Reservoir)
    fig = env._make_frame(step=0, action_idx=None, grav_ylim=grav_ylim, show_wells=show_wells)
    fig.savefig(os.path.join(save_dir, "frame_00.png"), bbox_inches='tight', dpi=dpi)
    plt.close(fig) # Close the figure to free up memory

    # Loop through actions and save subsequent frames
    for i, action_idx in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action_idx)

        fig = env._make_frame(step=i + 1, action_idx=action_idx, grav_ylim=grav_ylim, show_wells=show_wells)
        
        # Save the figure (e.g., frame_01.png, frame_02.png)
        filename = os.path.join(save_dir, f"frame_{i + 1:02d}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

        if terminated:
            print(f"*** LEAKAGE at step {i + 1} — stopping early ***")
            break
        if truncated:
            print(f"*** MAX STEPS reached at step {i + 1} ***")
            break
            
    print(f"Successfully saved images to the '{save_dir}' directory.")




# """
# Modified POMDP Carbon Storage Model - Multi-Region Injection (v2)

# Key Changes from v1:
# 1. CONFIGURABLE OBSERVATION SPACE: Choose timelapse gravity, borehole gravity, or both
# 2. DISCRETE ACTION SPACE: Matches the predefined ACTIONS list (injection rate tuples)
# 3. REWARD: (trapped CO2 - leaked CO2) in Mm³ per step
# 4. FLEXIBLE LEAKAGE: Episode does NOT end on leakage; max_steps=150
# 5. OPTIMIZED GRAVITY CACHING: Reduces computation overhead
# 6. FIXED SEED: Same geology for ALL episodes
# """

# import math
# import random
# from dataclasses import dataclass, asdict
# from typing import List, Tuple, Optional, Union, Dict
# import os

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
# from matplotlib import gridspec
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# # Try to import tqdm, use a fallback if not available
# try:
#     from tqdm import tqdm
# except ImportError:
#     def tqdm(iterable, **kwargs):
#         return iterable

# print("Imports done!")

# # ==============================================================================
# # DISCRETE ACTION SPACE DEFINITION
# # ==============================================================================
# # These are predefined injection rate tuples (region1, region2, region3)
# # Sum should be ~1,000,000 (total injection rate in m³/year distributed across regions)
# DISCRETE_ACTIONS = [
#     (0, 0, 999999.99), 
#     (0, 333333.33, 666666.66), 
#     (0, 666666.66, 333333.33), 
#     (0, 999999.99, 0), 
#     (333333.33, 0, 666666.66), 
#     (333333.33, 333333.33, 333333.33), 
#     (333333.33, 666666.66, 0), 
#     (666666.66, 0, 333333.33), 
#     (666666.66, 333333.33, 0), 
#     (999999.99, 0, 0)
# ]
# N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)

# # ==============================================================================
# # 1. GEOMETRY & PARAMETERS
# # ==============================================================================

# def h_top(x: np.ndarray, lobe: float, center: float, elev: float) -> np.ndarray:
#     """Calculates the structural elevation profile of the reservoir top (Caprock)."""
#     return lobe * np.sin(5 * np.pi * x) + center * np.sin(np.pi * x) + elev * x


# @dataclass
# class SubsurfaceParams:
#     """Data Class for storing the random parameters."""
#     lobe: float
#     center: float
#     elev: float
#     rho: float


# def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
#     """Generates random geological models."""
#     lobes = rng.uniform(0.05, 0.25, size=n)
#     centers = rng.uniform(0.05, 0.5, size=n)
#     elevs = rng.uniform(0.05, 0.5, size=n)
#     rhos = rng.uniform(0.5, 1.5, size=n)
#     return np.array([SubsurfaceParams(a, b, c, d) for a, b, c, d in zip(lobes, centers, elevs, rhos)])


# # ==============================================================================
# # 2. SCALING HELPERS
# # ==============================================================================

# class PhysicalScaler:
#     """Handles the conversion between Simulation Units and Physical Units."""
    
#     def __init__(self, x_len_m: float = 20000.0, res_thickness_m: float = 500.0, 
#                  slice_thickness_m: float = 50.0, top_depth_m: float = 2000.0):
#         self.x_scale = x_len_m
#         self.y_scale = res_thickness_m
#         self.z_thick = slice_thickness_m
#         self.top_depth = top_depth_m
#         self.base_depth = top_depth_m + res_thickness_m

#     def to_sim_x(self, x_meters: float) -> float:
#         return x_meters / self.x_scale

#     def to_sim_vol(self, vol_m3: float) -> float:
#         return vol_m3 / (self.x_scale * self.y_scale * self.z_thick)

#     def to_phys_x(self, sim_x: float) -> float:
#         return sim_x * self.x_scale

#     def to_phys_vol(self, sim_vol: float) -> float:
#         return sim_vol * (self.x_scale * self.y_scale * self.z_thick)

#     def to_phys_depth(self, sim_h: float) -> float:
#         return self.base_depth - (sim_h * self.y_scale)


# # ==============================================================================
# # 3. MAIN SIMULATION CLASS (SpillpointEnv)
# # ==============================================================================

# class SpillpointEnv:
#     """The Core Simulation Engine."""
    
#     def __init__(self, params: SubsurfaceParams,
#                  obs_well_loc_m: float = 11000.0,
#                  nx_cells: int = 100,
#                  nz_cells: int = 100,
#                  nx_sensors: int = 60,
#                  nz_sensors: int = 50):
#         self.params = params
#         self.obs_well_loc_m = obs_well_loc_m

#         self.nx_cells = nx_cells
#         self.nz_cells = nz_cells
#         self.nx_sensors = nx_sensors
#         self.nz_sensors = nz_sensors
        
#         # Initialize Scaler
#         # Original: reservoir top at 2000 m depth (weaker gravity signal at surface):
#         # self.scaler = PhysicalScaler(
#         #     x_len_m=20000.0,
#         #     res_thickness_m=500.0,
#         #     slice_thickness_m=50.0,
#         #     top_depth_m=2000.0
#         # )
#         # Shallow reservoir: top at 300 m depth (stronger gravity signal — sensors
#         # are closer to CO2, so timelapse gravity better resolves anticline shape):
#         self.scaler = PhysicalScaler(
#             x_len_m=20000.0,
#             res_thickness_m=500.0,
#             slice_thickness_m=50.0,
#             top_depth_m=300.0
#         )
        
#         # Generate Geometry
#         self.x = np.linspace(0.0, 1.0, self.nx_cells)
#         self.dx = self.x[1] - self.x[0]
#         self.h = h_top(self.x, params.lobe, params.center, params.elev)

#         self.h0 = float(self.h[0])
#         self.h1 = float(self.h[-1])

#         self.xv1, self.v1, self.xv3, self.v3 = self._find_two_minima()

#         self.mask1 = self.x <= self.xv1
#         self.mask2 = (self.x >= self.xv1) & (self.x <= self.xv3)
#         self.mask3 = self.x >= self.xv3

#         self.hmax1 = float(self.h[self.mask1].max())
#         self.hmax2 = float(self.h[self.mask2].max())
#         self.hmax3 = float(self.h[self.mask3].max())
#         self.hmax_all = float(self.h.max())

#         # Define Spill Thresholds
#         self.thresh1 = max(self.v1, self.h0)
#         self.thresh2 = max(self.v1, self.v3)
#         self.thresh3 = max(self.v3, self.h1)

#         self.top12, self.bot12 = self.v1, max(self.h0, self.v3)
#         self.top23, self.bot23 = self.v3, max(self.h1, self.v1)
#         self.top123, self.bot123 = min(self.v1, self.v3), max(self.h0, self.h1)

#         # Initialize Fluid State
#         self.v_reg = {1: 0.0, 2: 0.0, 3: 0.0}
#         self.v_shared_12 = 0.0
#         self.v_shared_23 = 0.0
#         self.v_shared_123 = 0.0
#         self.v_leaked = 0.0

#         self.cap1 = self._volume_region_at_level(self.mask1, self.thresh1)
#         self.cap2 = self._volume_region_at_level(self.mask2, self.thresh2)
#         self.cap3 = self._volume_region_at_level(self.mask3, self.thresh3)
#         self.cap12 = self._volume_between_levels(self.mask1 | self.mask2, self.top12, self.bot12)
#         self.cap23 = self._volume_between_levels(self.mask2 | self.mask3, self.top23, self.bot23)
#         self.cap123 = self._volume_between_levels(np.ones_like(self.x, bool), self.top123, self.bot123)

#         # Gravity Physics Setup
#         self.G = 6.67408e-11
#         self.rho_brine = 1000.0
#         self.rho_co2 = 650.0
#         self.density_diff = self.rho_co2 - self.rho_brine
#         self.si_to_ugal = 1e8
        
#         self._init_gravity_grid(nz_cells=nz_cells, ns_sensors=nx_sensors)
#         self._init_borehole_gravity(nz_sensors=nz_sensors)
        
#         # Store injector locations for each region (center of each region)
#         self.region_injector_locs_sim = self._compute_region_centers()

#     def _compute_region_centers(self) -> Dict[int, float]:
#         """Compute the center x-location for each of the 3 regions."""
#         center1 = 0.5 * self.xv1
#         center2 = 0.5 * (self.xv1 + self.xv3)
#         center3 = 0.5 * (self.xv3 + 1.0)
#         return {1: center1, 2: center2, 3: center3}

#     def get_region_injector_locations_m(self) -> Dict[int, float]:
#         """Get injector locations in physical units (meters)."""
#         return {
#             r: self.scaler.to_phys_x(loc) 
#             for r, loc in self.region_injector_locs_sim.items()
#         }

#     # ------------------------ Part A: Geometry Solvers ------------------------

#     def _find_two_minima(self) -> Tuple[float, float, float, float]:
#         y = self.h
#         idx = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
#         idx = idx[(idx > 0) & (idx < len(y) - 1)]
        
#         if len(idx) < 2:
#             interior = np.arange(1, len(y) - 1)
#             idx = interior[np.argsort(y[interior])[:2]]
#             idx.sort()
        
#         i1, i3 = int(idx[0]), int(idx[1])
#         return float(self.x[i1]), float(y[i1]), float(self.x[i3]), float(y[i3])

#     def _volume_region_at_level(self, mask: np.ndarray, level: float) -> float:
#         dh = np.clip(self.h[mask] - level, 0.0, None)
#         return float(self.params.rho * np.trapezoid(dh, self.x[mask]))

#     def _volume_between_levels(self, mask: np.ndarray, top: float, bottom: float) -> float:
#         if bottom >= top:
#             return 0.0
#         v_top = self._volume_region_at_level(mask, top)
#         v_bottom = self._volume_region_at_level(mask, bottom)
#         return float(v_bottom - v_top)

#     def _invert_volume_to_level(self, mask: np.ndarray, lo: float, hi: float, target: float) -> float:
#         V_lo = self._volume_region_at_level(mask, lo)
#         V_hi = self._volume_region_at_level(mask, hi)
#         target = float(np.clip(target, 0.0, V_lo))
        
#         if abs(target - V_lo) < 1e-12:
#             return lo
#         if target <= V_hi + 1e-12:
#             return hi
        
#         a, b = lo, hi
#         for _ in range(80):
#             m = 0.5 * (a + b)
#             Vm = self._volume_region_at_level(mask, m)
#             if Vm > target:
#                 a = m
#             else:
#                 b = m
#             if abs(Vm - target) < 1e-10 or abs(b - a) < 1e-10:
#                 return m
#         return 0.5 * (a + b)

#     # ------------------------ Part B: Public API ------------------------

#     def find_region(self, xloc: float) -> int:
#         if xloc <= self.xv1:
#             return 1
#         if xloc < self.xv3:
#             return 2
#         return 3

#     def calc_vol_rem(self, region: int) -> float:
#         return [None, self.cap1 - self.v_reg[1], self.cap2 - self.v_reg[2], self.cap3 - self.v_reg[3]][region]

#     def calc_vol_rem_shared_region(self, a: int, b: int) -> float:
#         s = {a, b}
#         if s == {1, 2}:
#             return self.cap12 - self.v_shared_12
#         if s == {2, 3}:
#             return self.cap23 - self.v_shared_23
#         return 0.0

#     def calc_rem_vol_all_shared_region(self) -> float:
#         return self.cap123 - self.v_shared_123

#     # ------------------------ Part C: Fluid Injection Logic ------------------------
    
#     def _next_after_full(self, region: int) -> tuple:
#         if region == 1:
#             return ('spill', 2) if (self.v1 > self.h0) else ('leak', None)
#         if region == 3:
#             return ('spill', 2) if (self.v3 > self.h1) else ('leak', None)
        
#         if self.v1 < self.v3:
#             return ('spill', 3)
#         if self.v1 > self.v3:
#             return ('spill', 1)
#         if self.h0 < self.h1:
#             return ('spill', 3)
#         if self.h0 > self.h1:
#             return ('spill', 1)
#         rem1, rem3 = self.calc_vol_rem(1), self.calc_vol_rem(3)
#         if rem1 < rem3:
#             return ('spill', 3)
#         if rem1 > rem3:
#             return ('spill', 1)
#         return ('spill', int(np.random.choice([1, 3])))

#     def inject(self, xloc: float, volume: float) -> dict:
#         """Inject CO2 at a specific location."""
#         EPS = 1e-12
#         cur = self.find_region(float(xloc))
#         S = {"to_r1": 0.0, "to_r2": 0.0, "to_r3": 0.0, "to_s12": 0.0, "to_s23": 0.0, "to_s123": 0.0, "leaked": 0.0}

#         rem_cur = max(0.0, self.calc_vol_rem(cur))
#         add = min(volume, rem_cur)
#         self.v_reg[cur] += add
#         S[f"to_r{cur}"] += add
#         left = volume - add
#         if left <= EPS:
#             return S

#         mode, adj = self._next_after_full(cur)
#         if mode == 'leak':
#             self.v_leaked += left
#             S["leaked"] += left
#             return S

#         rem_adj = max(0.0, self.calc_vol_rem(adj))
#         add = min(left, rem_adj)
#         self.v_reg[adj] += add
#         S[f"to_r{adj}"] += add
#         left -= add
#         if left <= EPS:
#             return S

#         def use_r(e, r):
#             if e <= 0:
#                 return 0.0
#             rem = max(0.0, self.calc_vol_rem(r))
#             add = min(e, rem)
#             self.v_reg[r] += add
#             S[f"to_r{r}"] += add
#             return e - add

#         def use_s12(e):
#             if e <= 0:
#                 return 0.0
#             rem = max(0.0, self.calc_vol_rem_shared_region(1, 2))
#             add = min(e, rem)
#             self.v_shared_12 += add
#             S["to_s12"] += add
#             return e - add

#         def use_s23(e):
#             if e <= 0:
#                 return 0.0
#             rem = max(0.0, self.calc_vol_rem_shared_region(2, 3))
#             add = min(e, rem)
#             self.v_shared_23 += add
#             S["to_s23"] += add
#             return e - add

#         def use_s123(e):
#             if e <= 0:
#                 return 0.0
#             rem = max(0.0, self.calc_rem_vol_all_shared_region())
#             add = min(e, rem)
#             self.v_shared_123 += add
#             S["to_s123"] += add
#             return e - add

#         pair = {cur, adj}
#         if pair == {1, 2}:
#             left = use_s12(left)
#             if left > EPS:
#                 if self.h0 > self.v3:
#                     pass
#                 else:
#                     left = use_r(left, 3)
#                     if left > EPS:
#                         left = use_s23(left)
#                     if left > EPS:
#                         left = use_s123(left)
#         elif pair == {2, 3}:
#             left = use_s23(left)
#             if left > EPS:
#                 if self.h1 > self.v1:
#                     pass
#                 else:
#                     left = use_r(left, 1)
#                     if left > EPS:
#                         left = use_s12(left)
#                     if left > EPS:
#                         left = use_s123(left)
#         else:
#             if adj == 2:
#                 left = use_s12(left) if cur == 1 else use_s23(left)
#                 if left > EPS:
#                     left = use_r(left, 1 if cur == 3 else 3)
#                 if left > EPS:
#                     left = use_s123(left)

#         if left > EPS:
#             self.v_leaked += left
#             S["leaked"] += left
#         return S

#     def inject_multi_region(self, volumes_per_region: Dict[int, float]) -> dict:
#         """
#         Inject CO2 into multiple regions simultaneously.
        
#         Args:
#             volumes_per_region: Dict mapping region (1,2,3) to volume to inject
        
#         Returns:
#             Combined result dict with totals
#         """
#         total_result = {"to_r1": 0.0, "to_r2": 0.0, "to_r3": 0.0, 
#                        "to_s12": 0.0, "to_s23": 0.0, "to_s123": 0.0, "leaked": 0.0}
        
#         for region, volume in volumes_per_region.items():
#             if volume > 0:
#                 xloc = self.region_injector_locs_sim[region]
#                 result = self.inject(xloc, volume)
#                 for key in total_result:
#                     total_result[key] += result[key]
        
#         return total_result

#     def is_valid_geometry(self, tol: float = 1e-8) -> bool:
#         y = self.h
#         idx = np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1
#         idx = idx[(idx > 0) & (idx < len(y) - 1)]
#         if len(idx) < 2:
#             return False
#         i1, i3 = int(idx[0]), int(idx[1])
#         if abs(self.x[i3] - self.x[i1]) < self.dx:
#             return False
#         v1, v3 = float(y[i1]), float(y[i3])
#         if (abs(v1 - self.h0) < tol) or (abs(v1 - self.h1) < tol):
#             return False
#         if (abs(v3 - self.h0) < tol) or (abs(v3 - self.h1) < tol):
#             return False
#         return True

#     # ------------------------ Part D: Gravity Engines ------------------------
    
#     def _init_gravity_grid(self, nz_cells=50, ns_sensors=60, sensor_x_range=(-0.4, 1.4)):
#         """Initializes surface gravity grid."""
#         self.grav_nz_cells = self.nz_cells
#         self.grav_nx_cells = self.nx_cells
        
#         phys_x = self.scaler.to_phys_x(self.x)
#         phys_dx = phys_x[1] - phys_x[0]
        
#         sim_h_max = self.hmax_all
#         sim_z_grid = np.linspace(0, sim_h_max, nz_cells)
#         sim_dz = sim_z_grid[1] - sim_z_grid[0]
        
#         phys_dz_cell = sim_dz * self.scaler.y_scale
#         self.grav_cell_volume_m3 = phys_dx * phys_dz_cell * self.scaler.z_thick

#         xx_sim, hh_sim = np.meshgrid(self.x, sim_z_grid)
#         flat_x_sim = xx_sim.ravel()
#         flat_h_sim = hh_sim.ravel()
        
#         self.grav_cell_x_m = self.scaler.to_phys_x(flat_x_sim)
#         self.grav_cell_z_m = self.scaler.to_phys_depth(flat_h_sim)
#         self.grav_cell_y_m = np.zeros_like(self.grav_cell_x_m)

#         N_cells = len(self.grav_cell_x_m)

#         sx_min, sx_max = sensor_x_range
#         self.grav_sensor_x_sim = np.linspace(sx_min, sx_max, ns_sensors)
#         sensor_x_m = self.scaler.to_phys_x(self.grav_sensor_x_sim)
#         sensor_z_m = np.zeros(ns_sensors)
#         sensor_y_m = np.zeros(ns_sensors)

#         self.grav_geo_matrix = np.zeros((N_cells, ns_sensors))
#         epsilon_sq = 1.0

#         for j in range(ns_sensors):
#             dx = self.grav_cell_x_m - sensor_x_m[j]
#             dy = self.grav_cell_y_m - sensor_y_m[j]
#             dz = self.grav_cell_z_m - sensor_z_m[j]
            
#             r_sq = dx**2 + dy**2 + dz**2 + epsilon_sq
#             r = np.sqrt(r_sq)
#             r_cubed = r**3
            
#             self.grav_geo_matrix[:, j] = dz / r_cubed

#         topo_h_at_cells = np.interp(flat_x_sim, self.x, self.h)
#         self.grav_is_reservoir_mask = (flat_h_sim < topo_h_at_cells)

#     def _init_borehole_gravity(self, nz_sensors=50):
#         """Initializes borehole gravity grid."""
#         self.bh_sensor_z_m = np.linspace(self.scaler.top_depth, self.scaler.base_depth, nz_sensors)
#         ns_bh = len(self.bh_sensor_z_m)
        
#         bh_x_m = self.obs_well_loc_m
#         bh_y_m = 0.0
#         N_cells = len(self.grav_cell_x_m)
        
#         self.borehole_geo_matrix_x = np.zeros((N_cells, ns_bh))
#         self.borehole_geo_matrix_z = np.zeros((N_cells, ns_bh))
#         epsilon_sq = 1.0
        
#         for j in range(ns_bh):
#             z_g = self.bh_sensor_z_m[j]
#             x_g = bh_x_m
            
#             diff_x = self.grav_cell_x_m - x_g
#             diff_y = self.grav_cell_y_m - bh_y_m
#             diff_z = self.grav_cell_z_m - z_g
            
#             r_sq = diff_x**2 + diff_y**2 + diff_z**2 + epsilon_sq
#             r = np.sqrt(r_sq)
#             r_cubed = r**3
            
#             self.borehole_geo_matrix_x[:, j] = diff_x / r_cubed
#             self.borehole_geo_matrix_z[:, j] = diff_z / r_cubed

#     def get_time_lapse_gravity(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Calculates SURFACE gravity (Vertical component only)."""
#         current_base_sim = self._compute_current_base()
#         cell_h_sim = (self.scaler.base_depth - self.grav_cell_z_m) / self.scaler.y_scale
#         cell_x_sim = self.grav_cell_x_m / self.scaler.x_scale
#         fluid_base_at_cells = np.interp(cell_x_sim, self.x, current_base_sim)
        
#         is_filled = self.grav_is_reservoir_mask & (cell_h_sim > fluid_base_at_cells)
        
#         mass_change = self.grav_cell_volume_m3 * self.params.rho * self.density_diff * is_filled.astype(float)
        
#         delta_g_SI = self.G * (mass_change @ self.grav_geo_matrix)
#         delta_g_SI = np.nan_to_num(delta_g_SI)
#         return self.scaler.to_phys_x(self.grav_sensor_x_sim), delta_g_SI * 1e8

#     def get_borehole_gravity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Calculates BOREHOLE gravity."""
#         current_base_sim = self._compute_current_base()
        
#         cell_h_sim = (self.scaler.base_depth - self.grav_cell_z_m) / self.scaler.y_scale
#         cell_x_sim = self.grav_cell_x_m / self.scaler.x_scale
#         fluid_base_at_cells = np.interp(cell_x_sim, self.x, current_base_sim)
        
#         is_filled = self.grav_is_reservoir_mask & (cell_h_sim > fluid_base_at_cells)
        
#         mass_change = self.grav_cell_volume_m3 * self.params.rho * self.density_diff * is_filled.astype(float)
        
#         delta_gx_SI = self.G * (mass_change @ self.borehole_geo_matrix_x)
#         delta_gx_uGal = np.nan_to_num(delta_gx_SI) * self.si_to_ugal
        
#         delta_gz_SI = self.G * (mass_change @ self.borehole_geo_matrix_z)
#         delta_gz_uGal = np.nan_to_num(delta_gz_SI) * self.si_to_ugal
        
#         return self.bh_sensor_z_m, delta_gx_uGal, delta_gz_uGal

#     # --------------------------------------------------------------------------
#     # PART 5: State computation and utilities
#     # --------------------------------------------------------------------------
    
#     def _level_region(self, r: int) -> float:
#         mask = [None, self.mask1, self.mask2, self.mask3][r]
#         thresh = [None, self.thresh1, self.thresh2, self.thresh3][r]
#         hmax = [None, self.hmax1, self.hmax2, self.hmax3][r]
#         return self._invert_volume_to_level(mask, thresh, hmax, self.v_reg[r])

#     def _level_s12(self) -> Optional[float]:
#         if self.v_shared_12 <= 0 or self.cap12 <= 0:
#             return None
#         mask = self.mask1 | self.mask2
#         Vtop = self._volume_region_at_level(mask, self.top12)
#         return self._invert_volume_to_level(mask, self.bot12, self.top12, self.v_shared_12 + Vtop)

#     def _level_s23(self) -> Optional[float]:
#         if self.v_shared_23 <= 0 or self.cap23 <= 0:
#             return None
#         mask = self.mask2 | self.mask3
#         Vtop = self._volume_region_at_level(mask, self.top23)
#         return self._invert_volume_to_level(mask, self.bot23, self.top23, self.v_shared_23 + Vtop)

#     def _level_s123(self) -> Optional[float]:
#         if self.v_shared_123 <= 0 or self.cap123 <= 0:
#             return None
#         mask = np.ones_like(self.x, bool)
#         Vtop = self._volume_region_at_level(mask, self.top123)
#         return self._invert_volume_to_level(mask, self.bot123, self.top123, self.v_shared_123 + Vtop)

#     def _compute_current_base(self) -> np.ndarray:
#         base = np.full_like(self.h, self.hmax_all + 1.0)
#         for r, mask in [(1, self.mask1), (2, self.mask2), (3, self.mask3)]:
#             d = self._level_region(r)
#             base[mask] = np.minimum(base[mask], d)
#         d12 = self._level_s12()
#         if d12 is not None:
#             base[self.mask1 | self.mask2] = np.minimum(base[self.mask1 | self.mask2], d12)
#         d23 = self._level_s23()
#         if d23 is not None:
#             base[self.mask2 | self.mask3] = np.minimum(base[self.mask2 | self.mask3], d23)
#         d123 = self._level_s123()
#         if d123 is not None:
#             base = np.minimum(base, d123)
#         base[self.mask1] = np.maximum(base[self.mask1], self.h0)
#         base[self.mask3] = np.maximum(base[self.mask3], self.h1)
#         return base

#     def trapped_volume(self) -> float:
#         return (self.v_reg[1] + self.v_reg[2] + self.v_reg[3]
#                 + self.v_shared_12 + self.v_shared_23 + self.v_shared_123)

#     def reset_fluid_state(self):
#         """Reset the fluid state to empty (for new episode with same geology)."""
#         self.v_reg = {1: 0.0, 2: 0.0, 3: 0.0}
#         self.v_shared_12 = 0.0
#         self.v_shared_23 = 0.0
#         self.v_shared_123 = 0.0
#         self.v_leaked = 0.0

#     def clone(self) -> 'SpillpointEnv':
#         new_env = SpillpointEnv(
#             self.params,
#             nx_cells=self.nx_cells,
#             obs_well_loc_m=self.obs_well_loc_m,
#             nz_cells=self.nz_cells,
#             nx_sensors=self.nx_sensors,
#             nz_sensors=self.nz_sensors
#         )
#         new_env.v_reg = self.v_reg.copy()
#         new_env.v_shared_12 = self.v_shared_12
#         new_env.v_shared_23 = self.v_shared_23
#         new_env.v_shared_123 = self.v_shared_123
#         new_env.v_leaked = self.v_leaked
#         new_env.h = self.h.copy()
#         new_env.xv1, new_env.v1, new_env.xv3, new_env.v3 = self.xv1, self.v1, self.xv3, self.v3
#         return new_env


#     # --------------------------------------------------------------------------
#     # PART 6: Visualization
#     # --------------------------------------------------------------------------

#     def plot_state_with_bars_and_gravity(self, step_title: str = "",
#                                           grav_ylim: Optional[Tuple[float, float]] = None,
#                                           injector_locs_m: Optional[Dict[int, float]] = None):
#         """
#         Four-panel visualization: cross-section, CO2 bar chart, surface gravity,
#         borehole gravity. Adapted from spillenv1.py for the v2 environment.
#         """
#         fig = plt.figure(figsize=(22, 5))
#         gs = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 2, 1], wspace=0.3)
#         ax = fig.add_subplot(gs[0, 0])
#         axb = fig.add_subplot(gs[0, 1])
#         axg = fig.add_subplot(gs[0, 2])
#         axbh = fig.add_subplot(gs[0, 3])

#         # 1. Cross Section
#         phys_x = self.scaler.to_phys_x(self.x)
#         phys_top = self.scaler.to_phys_depth(self.h)
#         phys_base_res = np.full_like(phys_x, self.scaler.base_depth)
#         sim_base = self._compute_current_base()
#         phys_fluid_contact = self.scaler.to_phys_depth(sim_base)

#         ax.plot(phys_x, phys_top, 'k', lw=2, label='Caprock')
#         fill_bottom = np.minimum(phys_fluid_contact, phys_base_res)
#         ax.fill_between(phys_x, phys_top, fill_bottom,
#                         where=(fill_bottom > phys_top),
#                         color="#4c78a8", alpha=0.8, label="CO₂ Plume")
#         ax.scatter([self.scaler.to_phys_x(self.xv1),
#                     self.scaler.to_phys_x(self.xv3)],
#                    [self.scaler.to_phys_depth(self.v1),
#                     self.scaler.to_phys_depth(self.v3)],
#                    color='red', marker='x', zorder=5, s=100, label="Spill Points")
#         ax.axvline(self.obs_well_loc_m, color='green', linestyle=':',
#                    linewidth=2, label="Obs Well")

#         if injector_locs_m:
#             # colors_inj = ['blue', 'orange', 'purple']
#             # for i, (region, loc_m) in enumerate(injector_locs_m.items()):
#             #     ax.axvline(loc_m, color=colors_inj[i], linestyle='--',
#             #               linewidth=2, label=f"Injector R{region}")
#             colors_inj = ['black', 'black', 'black']
#             for i, (region, loc_m) in enumerate(injector_locs_m.items()):
#                 ax.axvline(loc_m, color=colors_inj[i], linestyle='--',
#                           linewidth=2, label=f"Injector well")

#         ax.set_title(step_title)
#         ax.set_xlabel("Distance (m)")
#         ax.set_ylabel("Depth (m)")
#         ax.invert_yaxis()
#         ax.legend(loc="lower left", fontsize='small')
#         ax.grid(True, alpha=0.3)

#         # 2. Bar Chart
#         trapped_mm = self.scaler.to_phys_vol(self.trapped_volume()) / 1e6
#         leaked_mm = self.scaler.to_phys_vol(self.v_leaked) / 1e6
#         bars = axb.bar(["Trapped", "Leaked"], [trapped_mm, leaked_mm],
#                       color=["#4c78a8", "#d62728"], alpha=0.85)
#         axb.set_ylim(0, max((trapped_mm + leaked_mm) * 1.2, 10.0))
#         axb.set_title("CO₂ Balance")
#         axb.set_ylabel("Volume (Mm³)")
#         axb.grid(axis="y", alpha=0.3)
#         for bar, v in zip(bars, [trapped_mm, leaked_mm]):
#             axb.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
#                     f"{v:.2f}", ha='center', va='bottom', fontsize=10)

#         # 3. Surface Gravity
#         grav_x, grav_y = self.get_time_lapse_gravity()
#         axg.plot(grav_x, grav_y, 'm-o', markersize=4)
#         axg.set_title("Time-Lapse Gravity")
#         axg.set_xlabel("Position (m)")
#         axg.set_ylabel(r"$\Delta g$ ($\mu$Gal)")
#         axg.grid(True)
#         if grav_ylim:
#             axg.set_ylim(grav_ylim)

#         # 4. Borehole Gravity
#         bh_z, bh_gx, bh_gz = self.get_borehole_gravity()
#         axbh.plot(bh_gz, bh_z, 'g-o', markersize=4, label=r'$g_z$ (Vert)')
#         axbh.plot(bh_gx, bh_z, 'orange', linestyle='--', marker='x',
#                   markersize=4, label=r'$g_x$ (Horiz)')
#         axbh.set_title("Borehole Gravity")
#         axbh.set_xlabel(r"$\Delta g$ ($\mu$Gal)")
#         axbh.set_ylabel("Depth (m)")
#         axbh.invert_yaxis()
#         axbh.grid(True)
#         axbh.legend(fontsize='x-small')

#         plt.tight_layout()
#         return fig, (ax, axb, axg, axbh)

#     def plot_cross_section(self, ax=None, title: str = "",
#                            injector_locs_m: Optional[Dict[int, float]] = None,
#                            show_regions: bool = True):
#         """
#         Plot just the cross-section with CO2 fill and region shading.
#         Lighter-weight than the full 4-panel plot, suitable for GIF frames.
#         """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(12, 5))
#         else:
#             fig = ax.get_figure()

#         phys_x = self.scaler.to_phys_x(self.x)
#         phys_top = self.scaler.to_phys_depth(self.h)
#         phys_base_res = np.full_like(phys_x, self.scaler.base_depth)
#         sim_base = self._compute_current_base()
#         phys_fluid_contact = self.scaler.to_phys_depth(sim_base)

#         # Shade the three regions with light background colours
#         if show_regions:
#             region_colors = ['#cce5ff', '#d4edda', '#fff3cd']  # light blue, green, yellow
#             region_labels = ['Region 1', 'Region 2', 'Region 3']
#             for mask, col, lbl in zip(
#                     [self.mask1, self.mask2, self.mask3],
#                     region_colors, region_labels):
#                 ax.fill_between(phys_x, phys_top, phys_base_res,
#                                 where=mask, color=col, alpha=0.35, label=lbl)

#         # CO2 plume
#         fill_bottom = np.minimum(phys_fluid_contact, phys_base_res)
#         ax.fill_between(phys_x, phys_top, fill_bottom,
#                         where=(fill_bottom > phys_top),
#                         color="#4c78a8", alpha=0.85, label="CO₂ Plume")

#         # Caprock line on top
#         ax.plot(phys_x, phys_top, 'k', lw=2, label='Caprock')

#         # Spillpoints
#         ax.scatter([self.scaler.to_phys_x(self.xv1),
#                     self.scaler.to_phys_x(self.xv3)],
#                    [self.scaler.to_phys_depth(self.v1),
#                     self.scaler.to_phys_depth(self.v3)],
#                    color='red', marker='x', zorder=5, s=120, linewidths=2.5,
#                    label="Spill Points")

#         # Region boundary lines
#         for xsim in [self.xv1, self.xv3]:
#             ax.axvline(self.scaler.to_phys_x(xsim), color='grey',
#                        linestyle=':', linewidth=1, alpha=0.6)

#         # Injectors
#         if injector_locs_m:
#             # inj_colors = ['#0d6efd', '#fd7e14', '#6f42c1']
#             inj_colors = ['black', 'black', 'black']
#             for i, (region, loc_m) in enumerate(injector_locs_m.items()):
#                 # ax.axvline(loc_m, color=inj_colors[i], linestyle='--',
#                 #           linewidth=1.8, label=f"Injector R{region}")
#                 ax.axvline(loc_m, color=inj_colors[i], linestyle='--',
#                           linewidth=1.8, label="Injector well")

#         ax.set_xlabel("Distance (m)")
#         ax.set_ylabel("Depth (m)")
#         ax.set_title(title)
#         ax.invert_yaxis()
#         ax.legend(loc="lower left", fontsize='small', ncol=2)
#         ax.grid(True, alpha=0.2)

#         return fig, ax


# def build_valid_env(start_seed: int,
#                     max_tries: int = 500,
#                     obs_well_loc: float = 11000.0,
#                     nx_cells: int = 100,
#                     nz_cells: int = 100,
#                     nx_sensors: int = 60,
#                     nz_sensors: int = 50) -> SpillpointEnv:
#     """Build a SpillpointEnv with valid geometry."""
#     seed = start_seed
#     for _ in range(max_tries):
#         rng = np.random.default_rng(seed)
#         np.random.seed(seed)
#         params = sample_prior(1, rng=rng)[0]
        
#         env = SpillpointEnv(params, obs_well_loc_m=obs_well_loc,
#                            nx_cells=nx_cells, nz_cells=nz_cells,
#                            nx_sensors=nx_sensors, nz_sensors=nz_sensors)
        
#         if env.is_valid_geometry():
#             print(f"Found valid geometry with seed: {seed}")
#             return env
#         seed += 1
#     raise RuntimeError("Failed to find valid geometry.")


# print("SpillpointEnv code complete!")


# # ==============================================================================
# # 4. GYMNASIUM ENVIRONMENT - MULTI-REGION INJECTION (v2)
# # ==============================================================================

# try:
#     import gymnasium as gym
#     from gymnasium import spaces
#     GYM_AVAILABLE = True
# except ImportError:
#     try:
#         import gym
#         from gym import spaces
#         GYM_AVAILABLE = True
#     except ImportError:
#         GYM_AVAILABLE = False
#         print("Warning: Neither gymnasium nor gym is available. Using fallback base class.")
        
#         class FallbackEnv:
#             metadata = {}
#             observation_space = None
#             action_space = None
            
#             def reset(self, seed=None, options=None):
#                 pass
            
#             def step(self, action):
#                 pass
            
#             def render(self):
#                 pass
            
#             def close(self):
#                 pass
        
#         class spaces:
#             class Box:
#                 def __init__(self, low, high, shape=None, dtype=np.float32):
#                     self.low = np.array(low) if not isinstance(low, np.ndarray) else low
#                     self.high = np.array(high) if not isinstance(high, np.ndarray) else high
#                     self.shape = shape if shape else self.low.shape
#                     self.dtype = dtype
                
#                 def sample(self):
#                     return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
            
#             class Discrete:
#                 def __init__(self, n):
#                     self.n = n
                
#                 def sample(self):
#                     return np.random.randint(0, self.n)
        
#         gym = type('gym', (), {'Env': FallbackEnv})()

# if GYM_AVAILABLE:
#     BaseEnvClass = gym.Env
# else:
#     BaseEnvClass = FallbackEnv


# class MultiRegionCO2StorageEnv(BaseEnvClass):
#     """
#     Multi-Region CO2 Storage Environment (v2)
    
#     Key Features:
#     1. FIXED SEED - Same geology for ALL episodes (seed doesn't change)
#     2. 3 injector locations (one per region)
#     3. DISCRETE ACTION SPACE - Agent selects from predefined injection rate tuples
#     4. CONFIGURABLE OBSERVATIONS - Choose timelapse gravity, borehole gravity, or both
#     5. REWARD: (trapped CO2 - leaked CO2) in Mm³ per step
#     6. Leakage does NOT terminate the episode; max_steps=150 truncation only
    
#     Action Space: Discrete(N_DISCRETE_ACTIONS)
#         - Each action corresponds to a predefined (r1, r2, r3) injection rate tuple
        
#     Observation Space: Depends on obs_mode parameter
#         - 'timelapse': Only time-lapse gravity measurements (nx_sensors,)
#         - 'borehole': Only borehole gravity measurements (nz_sensors * 2,) [gz and gx]
#         - 'both': Both measurements concatenated (nx_sensors + nz_sensors * 2,)
#     """

#     metadata = {"render_modes": ["human", "rgb_array"]}

#     def __init__(self,
#                  fixed_seed: int = 42,
#                  obs_mode: str = 'both',  # 'timelapse', 'borehole', or 'both'
#                  obs_well_loc_m: float = 10000.0,
#                  nx_cells: int = 100,
#                  nz_cells: int = 100,
#                  nx_sensors: int = 30,
#                  nz_sensors: int = 30,
#                  max_steps: int = 150,
#                  render_mode: Optional[str] = None):
        
#         super(MultiRegionCO2StorageEnv, self).__init__()

#         # --- FIXED seed - same geology for ALL episodes ---
#         self.fixed_seed = fixed_seed
        
#         # --- Observation mode ---
#         assert obs_mode in ['timelapse', 'borehole', 'both'], \
#             f"obs_mode must be 'timelapse', 'borehole', or 'both', got '{obs_mode}'"
#         self.obs_mode = obs_mode
        
#         self.obs_well_loc = obs_well_loc_m
#         self.max_steps = max_steps
#         self.render_mode = render_mode
        
#         # Simulation config
#         self.sim_config = {
#             'nx_cells': nx_cells,
#             'nz_cells': nz_cells,
#             'nx_sensors': nx_sensors,
#             'nz_sensors': nz_sensors
#         }
#         self.nx_sensors = nx_sensors
#         self.nz_sensors = nz_sensors

#         # --- DISCRETE Action Space ---
#         # Agent selects an index into DISCRETE_ACTIONS list
#         self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

#         # --- Observation Space (POMDP - only gravity measurements) ---
#         if obs_mode == 'timelapse':
#             obs_size = nx_sensors
#         elif obs_mode == 'borehole':
#             obs_size = nz_sensors * 2  # gz and gx components
#         else:  # 'both'
#             obs_size = nx_sensors + nz_sensors * 2
        
#         self.observation_space = spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(obs_size,),
#             dtype=np.float32
#         )
#         self.obs_size = obs_size

#         # Build simulation environment ONCE with fixed seed
#         print(f"Building environment with FIXED seed: {self.fixed_seed}")
#         print(f"Observation mode: {self.obs_mode} (obs_size={obs_size})")
#         self.sim_env = build_valid_env(
#             start_seed=self.fixed_seed,
#             obs_well_loc=self.obs_well_loc,
#             **self.sim_config
#         )
        
#         # Store injector locations
#         self.injector_locs_m = self.sim_env.get_region_injector_locations_m()
#         print(f"Injector locations: R1={self.injector_locs_m[1]:.0f}m, "
#               f"R2={self.injector_locs_m[2]:.0f}m, R3={self.injector_locs_m[3]:.0f}m")
        
#         # Internal state
#         self.current_step = 0
#         self.total_trapped_m3 = 0.0
#         self.total_leaked_m3 = 0.0
#         self.region_trapped_m3 = {1: 0.0, 2: 0.0, 3: 0.0}
        
#         # Cache for gravity (computed once per step)
#         self._cached_gravity = None
#         self._gravity_step = -1

#     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#         """
#         Reset the environment for a new episode.
        
#         NOTE: The geology stays the SAME (fixed seed). Only fluid state resets.
#         """
#         super().reset(seed=seed)
        
#         # Reset fluid state (but keep same geology!)
#         self.sim_env.reset_fluid_state()
        
#         # Reset counters
#         self.current_step = 0
#         self.total_trapped_m3 = 0.0
#         self.total_leaked_m3 = 0.0
#         self.region_trapped_m3 = {1: 0.0, 2: 0.0, 3: 0.0}
        
#         # Clear gravity cache
#         self._cached_gravity = None
#         self._gravity_step = -1
        
#         # Get initial observation
#         observation = self._get_observation()
        
#         info = {
#             "fixed_seed": self.fixed_seed,
#             "obs_mode": self.obs_mode,
#             "injector_locations_m": self.injector_locs_m,
#             "trapped_total_m3": 0.0,
#             "leaked_total_m3": 0.0
#         }
        
#         return observation, info

#     def step(self, action: int):
#         """
#         Execute one time step using discrete action.
        
#         Args:
#             action: int - Index into DISCRETE_ACTIONS list
        
#         Returns:
#             observation, reward, terminated, truncated, info
#         """
#         # 1. Get injection volumes from action
#         action_tuple = DISCRETE_ACTIONS[action]  # (r1_vol, r2_vol, r3_vol)
        
#         # 2. Convert to simulation units and inject
#         scaler = self.sim_env.scaler
#         volumes_sim = {}
#         for region in [1, 2, 3]:
#             vol_m3 = action_tuple[region - 1]  # Already in m³
#             volumes_sim[region] = scaler.to_sim_vol(vol_m3)
        
#         # 3. Inject CO2 into all regions
#         result_dict = self.sim_env.inject_multi_region(volumes_sim)
        
#         # 4. Calculate trapped and leaked volumes
#         trapped_sim = sum(v for k, v in result_dict.items() if k != 'leaked')
#         leaked_sim = result_dict.get('leaked', 0.0)
        
#         trapped_m3 = scaler.to_phys_vol(trapped_sim)
#         leaked_m3 = scaler.to_phys_vol(leaked_sim)
        
#         # Track per-region trapped
#         for region in [1, 2, 3]:
#             region_trapped = scaler.to_phys_vol(result_dict.get(f'to_r{region}', 0.0))
#             self.region_trapped_m3[region] += region_trapped
        
#         self.total_trapped_m3 += trapped_m3
#         self.total_leaked_m3 += leaked_m3
        
#         # 5. Check for leakage (no longer terminates episode)
#         has_leaked = leaked_m3 > 1e-3
        
#         # 6. OLD SIMPLIFIED REWARD: +1 for no leakage, 0 for leakage
#         # if has_leaked:
#         #     step_reward = 0.0
#         #     terminated = True
#         # else:
#         #     step_reward = 1.0
#         #     terminated = False
        
#         # 6. NEW REWARD: (trapped CO2 - leaked CO2) in Mm³
#         #    Leakage no longer terminates the episode — agent must learn to avoid it.
#         step_reward = (trapped_m3 - leaked_m3) / 1e6
#         terminated = False
        
#         # 7. Check truncation
#         self.current_step += 1
#         truncated = self.current_step >= self.max_steps
        
#         # 8. Invalidate gravity cache (state changed)
#         self._gravity_step = -1
        
#         # 9. Get observation
#         observation = self._get_observation()
        
#         # 10. Build info dict
#         info = {
#             "step": self.current_step,
#             "action_idx": action,
#             "action_tuple": action_tuple,
#             "trapped_step_m3": trapped_m3,
#             "leaked_step_m3": leaked_m3,
#             "trapped_total_m3": self.total_trapped_m3,
#             "leaked_total_m3": self.total_leaked_m3,
#             "has_leaked": has_leaked,
#             "region_trapped_m3": self.region_trapped_m3.copy()
#         }
        
#         return observation, float(step_reward), terminated, truncated, info

#     def _get_observation(self) -> np.ndarray:
#         """Construct observation vector with gravity measurements only (POMDP)."""
#         # Use cached gravity if available for current step
#         if self._gravity_step != self.current_step:
#             # Compute and cache gravity measurements
#             _, surf_grav = self.sim_env.get_time_lapse_gravity()
#             _, bh_gx, bh_gz = self.sim_env.get_borehole_gravity()
#             self._cached_gravity = (surf_grav, bh_gz, bh_gx)
#             self._gravity_step = self.current_step
        
#         surf_grav, bh_gz, bh_gx = self._cached_gravity
        
#         # Build observation based on mode
#         if self.obs_mode == 'timelapse':
#             obs = surf_grav
#         elif self.obs_mode == 'borehole':
#             obs = np.concatenate([bh_gz, bh_gx])
#         else:  # 'both'
#             obs = np.concatenate([surf_grav, bh_gz, bh_gx])
        
#         return obs.astype(np.float32)

#     def render(self):
#         """Render current state."""
#         if self.sim_env is None:
#             return None
            
#         # title = (f"Step {self.current_step} | "
#         #         f"Trapped: {self.total_trapped_m3/1e6:.2f} Mm³ | "
#         #         f"Leaked: {self.total_leaked_m3/1e6:.4f} Mm³")
#         title = (f"Trapped: {self.total_trapped_m3/1e6:.2f} Mm³ | "
#                 f"Leaked: {self.total_leaked_m3/1e6:.4f} Mm³")
        
#         fig, _ = self.sim_env.plot_state_with_bars_and_gravity(
#             title,
#             injector_locs_m=self.injector_locs_m
#         )
        
#         if self.render_mode == "rgb_array":
#             fig.canvas.draw()
#             data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#             data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             plt.close(fig)
#             return data
#         else:
#             return fig

#     # ------------------------------------------------------------------
#     # Visualization: replay a list of actions as static plots or GIF
#     # ------------------------------------------------------------------

#     def _make_frame(self, step: int, action_idx: Optional[int],
#                     grav_ylim: Optional[Tuple[float, float]] = None,
#                     show_wells: Optional[List[int]] = None,
#                     figsize: Tuple[float, float] = (18, 8)):
#         """
#         Produce a single figure for the current environment state.
        
#         Layout (2 rows, equal height):
#             Top row:  spillpoint cross-section (depth 0 to base_depth)
#             Bottom row: timelapse gravity (full width)
        
#         Args:
#             step: current step number (0 = initial empty state)
#             action_idx: the discrete action taken to reach this state (None for step 0)
#             grav_ylim: optional fixed y-limits for gravity panel (keeps GIF stable)
#             show_wells: list of region numbers whose injector wells to show, e.g. [1,2,3].
#                         Default [1,2,3]. Pass [] to hide all wells.
#             figsize: figure size
            
#         Returns:
#             matplotlib Figure
#         """
#         if show_wells is None:
#             show_wells = [1, 2, 3]

#         sim = self.sim_env
#         scaler = sim.scaler

#         fig = plt.figure(figsize=figsize, facecolor='white')
#         gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.45)

#         # ---- Top: cross-section (full width) ----
#         ax_cs = fig.add_subplot(gs_main[0])
#         ax_cs.set_facecolor('white')

#         phys_x = scaler.to_phys_x(sim.x)
#         phys_top = scaler.to_phys_depth(sim.h)
#         phys_base_res = np.full_like(phys_x, scaler.base_depth)
#         sim_base = sim._compute_current_base()
#         phys_fluid = scaler.to_phys_depth(sim_base)

#         # Original code start
#         # # Brown overburden rock: fills between caprock (top surface) and
#         # # depth 0 (ground surface), only within x=[0, 20000].
#         # # This represents rock layers above the caprock but below the surface.
#         # ax_cs.fill_between(phys_x, 0, phys_top,
#         #                    color='#8B6914', alpha=0.70, label='Overburden')
#         # Original code end

#         # New code start
#         # Extended Overburden
#         ext_x = np.concatenate([[-6000], phys_x, [scaler.x_scale + 6000]])
#         ext_top = np.concatenate([[float(phys_top[0])], phys_top, [float(phys_top[-1])]])
#         ax_cs.fill_between(ext_x, 0, ext_top,
#                            color='#8B6914', alpha=0.70, label='Overburden')
#         # New code end

#         # Uniform light-blue brine fill (replaces per-region colors)
#         # ax_cs.fill_between(phys_x, phys_top, phys_base_res,
#         #                    color='#ADD8E6', alpha=0.60, label='Brine')
#         ax_cs.fill_between(phys_x, phys_top, phys_base_res,
#                            where=(phys_base_res > phys_top),
#                            color='#ADD8E6', alpha=0.60, label='Brine')

#         # CO2 plume (orange)
#         fill_bottom = np.minimum(phys_fluid, phys_base_res)
#         ax_cs.fill_between(phys_x, phys_top, fill_bottom,
#                            where=(fill_bottom > phys_top),
#                            color='#FF8C00', alpha=0.90, label="CO₂ Plume")

#         # Caprock — gray, thick (anticline line only, no base line)
#         ax_cs.plot(phys_x, phys_top, color='gray', lw=5, label='Caprock',
#                    zorder=4, solid_capstyle='round')

#         # Spillpoints
#         ax_cs.scatter(
#             [scaler.to_phys_x(sim.xv1), scaler.to_phys_x(sim.xv3)],
#             [scaler.to_phys_depth(sim.v1), scaler.to_phys_depth(sim.v3)],
#             color='red', marker='x', zorder=5, s=120, linewidths=2.5,
#             label="Spill Points")

#         # Region boundary lines
#         for xsim in [sim.xv1, sim.xv3]:
#             ax_cs.axvline(scaler.to_phys_x(xsim), color='grey',
#                           linestyle=':', linewidth=1, alpha=0.6)
            

#         # ---------- 100x100 Simulation Grid ----------
#         # Calculate boundaries based on the scaler
#         x_start, x_end = 0, scaler.x_scale
#         y_start, y_end = 0, scaler.base_depth
        
#         # Generate the coordinates for 100 cells (101 lines)
#         x_lines = np.linspace(x_start, x_end, 101)
#         y_lines = np.linspace(y_start, y_end, 101)
        
#         # Draw vertical lines strictly between y=0 and y=800
#         ax_cs.vlines(x_lines, ymin=y_start, ymax=y_end, 
#                      color='black', linewidth=0.3, alpha=0.5, zorder=6)
            
#         # Draw horizontal lines strictly between x=0 and x=20000
#         ax_cs.hlines(y_lines, xmin=x_start, xmax=x_end, 
#                      color='black', linewidth=0.3, alpha=0.5, zorder=6)

#         # ---------- Faults at x=0 and x=20000 ----------
#         # Each fault starts at the actual reservoir boundary depth at its
#         # respective edge, and extends outward at 45° going upward by 50m.
#         # fault_dy = 50.0   # vertical span (going upward)
#         fault_dy = 250.0   # vertical span (going upward)
#         fault_dx = 250.0   # horizontal span (45°)
#         # Left fault: starts at depth of caprock at x=0
#         left_fault_depth = float(phys_top[0])   # actual depth at x=0
#         ax_cs.plot([0, -fault_dx], [left_fault_depth, left_fault_depth - fault_dy],
#                    color='black', lw=2.5, solid_capstyle='round')
#         for frac in [0.3, 0.6]:
#             fx = -fault_dx * frac
#             fy = left_fault_depth - fault_dy * frac
#             ax_cs.plot([fx, fx - 15], [fy, fy + 10],
#                        color='black', lw=1.5, solid_capstyle='round')
#         # Right fault: starts at depth of caprock at x=20000
#         x_right = scaler.x_scale
#         right_fault_depth = float(phys_top[-1])  # actual depth at x=20000
#         ax_cs.plot([x_right, x_right + fault_dx],
#                    [right_fault_depth, right_fault_depth - fault_dy],
#                    color='black', lw=2.5, solid_capstyle='round')
#         for frac in [0.3, 0.6]:
#             fx = x_right + fault_dx * frac
#             fy = right_fault_depth - fault_dy * frac
#             ax_cs.plot([fx, fx + 15], [fy, fy + 10],
#                        color='black', lw=1.5, solid_capstyle='round')

#         # ---------- Surface sensor line at depth 0 ----------
#         sensor_x_phys = scaler.to_phys_x(sim.grav_sensor_x_sim)
#         ax_cs.axhline(0, color='gray', lw=1.0, alpha=0.5)
#         ax_cs.scatter(sensor_x_phys,
#                       np.zeros(len(sensor_x_phys)),
#                       color='green', marker='+', s=80, linewidths=1.8,
#                       zorder=6, label='Gravity Sensors')

#         # # Injector well lines (only for wells in show_wells)
#         # inj_colors = {1: '#0d6efd', 2: '#fd7e14', 3: '#6f42c1'}
#         # for region in show_wells:
#         #     if region in self.injector_locs_m:
#         #         loc_m = self.injector_locs_m[region]
#         #         ax_cs.axvline(loc_m, color=inj_colors[region], linestyle='--',
#         #                      linewidth=1.8, label=f"Injector R{region}")
                
#         inj_colors = {1: '#000000', 2: '#000000', 3: '#000000'}
#         added_label = False
#         for region in show_wells:
#             if region in self.injector_locs_m:
#                 loc_m = self.injector_locs_m[region]
#                 current_label = "Injector well" if not added_label else None
#                 ax_cs.axvline(loc_m, color=inj_colors[region], linestyle='--',
#                              linewidth=1.8, label=current_label)
#                 added_label = True

#         # Build title
#         trapped_mm3 = self.total_trapped_m3 / 1e6
#         leaked_mm3 = self.total_leaked_m3 / 1e6
#         total_cap_mm3 = scaler.to_phys_vol(
#             sim.cap1 + sim.cap2 + sim.cap3 + sim.cap12 + sim.cap23 + sim.cap123) / 1e6
#         if step == 0:
#             title = (f"Step 0 — Empty Reservoir  |  "
#                      f"Total Capacity: {total_cap_mm3:.2f} Mm³")
#         else:
#             act_tuple = DISCRETE_ACTIONS[action_idx]
#             total_rate = sum(act_tuple)
#             fracs = tuple(v / total_rate for v in act_tuple)
#             # title = (f"Step {step}  |  Action {action_idx}: "
#             #          f"R1={fracs[0]:.0%} R2={fracs[1]:.0%} R3={fracs[2]:.0%}  |  "
#             #          f"Trapped: {trapped_mm3:.2f} / {total_cap_mm3:.2f} Mm³  |  "
#             #          f"Leaked: {leaked_mm3:.4f} Mm³")
            
#             # New code
#             # Calculate the fill percentage for each region
#             r1_fill = (sim.v_reg[1] / sim.cap1) if sim.cap1 > 0 else 0
#             r2_fill = (sim.v_reg[2] / sim.cap2) if sim.cap2 > 0 else 0
#             r3_fill = (sim.v_reg[3] / sim.cap3) if sim.cap3 > 0 else 0
            
#             title = (f"Step {step}  |  Filled: "
#                      f"R1={r1_fill:.0%} R2={r2_fill:.0%} R3={r3_fill:.0%}  |  "
#                      f"Trapped: {trapped_mm3:.2f} / {total_cap_mm3:.2f} Mm³  |  "
#                      f"Leaked: {leaked_mm3:.4f} Mm³")

#         ax_cs.set_title(title, fontsize=12, fontweight='bold')
#         ax_cs.set_xlabel("Distance (m)")
#         ax_cs.set_ylabel("Depth (m)")
#         # y-axis: surface at top, base at bottom; padding above for sensors,
#         # below for faults
#         ax_cs.set_ylim(scaler.base_depth + 30, -60)
#         # ax_cs.set_xlim(-6000, scaler.x_scale + 6000)
#         ax_cs.set_xlim(0, scaler.x_scale)
#         # ax_cs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
#         #              fontsize='small', ncol=4, framealpha=0.9)
#         # ax_cs.legend(loc="center left", fontsize='small', ncol=2, framealpha=0.9)
#         ax_cs.grid(True, alpha=0.15)

#         # # ---- Bottom: timelapse gravity (full width) ----
#         # ax_grav = fig.add_subplot(gs_main[1])
#         # ax_grav.set_facecolor('white')

#         # grav_x, grav_y = sim.get_time_lapse_gravity()
#         # ax_grav.plot(grav_x, grav_y, 'm-o', markersize=3, linewidth=1.5)
#         # # Mark region boundaries on gravity plot
#         # for xsim in [sim.xv1, sim.xv3]:
#         #     ax_grav.axvline(scaler.to_phys_x(xsim), color='grey',
#         #                     linestyle=':', linewidth=0.8, alpha=0.5)
#         # ax_grav.set_title("Time-Lapse Gravity")
#         # ax_grav.set_xlabel("Position (m)")
#         # ax_grav.set_ylabel(r"$\Delta g$ ($\mu$Gal)")
#         # ax_grav.grid(True, alpha=0.3)
#         # # Match x-limits with cross-section for direct comparison
#         # ax_grav.set_xlim(-6000, scaler.x_scale + 6000)
#         # if grav_ylim:
#         #     ax_grav.set_ylim(grav_ylim)

#         return fig

#     def visualize_actions(self, actions: List[int],
#                           grav_ylim: Optional[Tuple[float, float]] = None,
#                           show_wells: Optional[List[int]] = None,
#                           figsize: Tuple[float, float] = (18, 8)):
#         """
#         Replay a list of discrete actions and display static inline plots
#         (designed for Jupyter notebooks — calls plt.show() for each frame).

#         Shows an initial empty-reservoir frame (step 0) followed by one frame
#         per action.  Stops early if leakage occurs.

#         Args:
#             actions: list of discrete action indices to replay
#             grav_ylim: optional fixed (ymin, ymax) for gravity panels.
#                        If None, automatically pre-computed from a dry run.
#             show_wells: list of region numbers whose injector wells to show,
#                         e.g. [1,2,3] (default). Pass [] to hide all.
#             figsize: per-frame figure size

#         Returns:
#             list of info dicts (one per executed step)
#         """
#         if show_wells is None:
#             show_wells = [1, 2, 3]

#         # --- Pre-compute static gravity y-limits if not provided ---
#         if grav_ylim is None:
#             self.reset()
#             grav_min, grav_max = 0.0, 0.0
#             for action_idx in actions:
#                 obs, reward, terminated, truncated, info = self.step(action_idx)
#                 _, grav_y = self.sim_env.get_time_lapse_gravity()
#                 grav_min = min(grav_min, float(grav_y.min()))
#                 grav_max = max(grav_max, float(grav_y.max()))
#                 if terminated or truncated:
#                     break
#             pad = max(abs(grav_max - grav_min) * 0.10, 0.5)
#             grav_ylim = (grav_min - pad, grav_max + pad)

#         # Reset to empty state
#         self.reset()

#         # Step 0: empty reservoir
#         fig = self._make_frame(step=0, action_idx=None,
#                                grav_ylim=grav_ylim, show_wells=show_wells,
#                                figsize=figsize)
#         plt.show()

#         infos = []
#         for i, action_idx in enumerate(actions):
#             obs, reward, terminated, truncated, info = self.step(action_idx)
#             infos.append(info)

#             fig = self._make_frame(step=i + 1, action_idx=action_idx,
#                                    grav_ylim=grav_ylim, show_wells=show_wells,
#                                    figsize=figsize)
#             plt.show()

#             if terminated:
#                 print(f"*** LEAKAGE at step {i + 1} — episode terminated ***")
#                 break
#             if truncated:
#                 print(f"*** MAX STEPS reached at step {i + 1} ***")
#                 break

#         return infos

#     def create_gif(self, actions: List[int],
#                    save_path: str = "co2_filling.gif",
#                    fps: int = 2,
#                    grav_ylim: Optional[Tuple[float, float]] = None,
#                    show_wells: Optional[List[int]] = None,
#                    figsize: Tuple[float, float] = (18, 8),
#                    dpi: int = 100):
#         """
#         Replay a list of discrete actions and save an animated GIF.

#         The GIF starts with the empty reservoir and adds one frame per
#         action, stopping early on leakage.

#         Args:
#             actions: list of discrete action indices
#             save_path: output file path for the GIF
#             fps: frames per second
#             grav_ylim: optional fixed y-limits for gravity panels.
#                        If None, automatically pre-computed from a dry run.
#             show_wells: list of region numbers whose injector wells to show,
#                         e.g. [1,2,3] (default). Pass [] to hide all.
#             figsize: figure size per frame
#             dpi: resolution

#         Returns:
#             list of info dicts for each executed step
#         """
#         import io
#         from PIL import Image

#         if show_wells is None:
#             show_wells = [1, 2, 3]

#         # --- Pre-compute static gravity y-limits if not provided ---
#         if grav_ylim is None:
#             self.reset()
#             grav_min, grav_max = 0.0, 0.0
#             for action_idx in actions:
#                 obs, reward, terminated, truncated, info = self.step(action_idx)
#                 _, grav_y = self.sim_env.get_time_lapse_gravity()
#                 grav_min = min(grav_min, float(grav_y.min()))
#                 grav_max = max(grav_max, float(grav_y.max()))
#                 if terminated or truncated:
#                     break
#             pad = max(abs(grav_max - grav_min) * 0.10, 0.5)
#             grav_ylim = (grav_min - pad, grav_max + pad)

#         frames: List[Image.Image] = []
#         infos = []

#         # Reset
#         self.reset()

#         # Frame 0: empty
#         fig = self._make_frame(step=0, action_idx=None,
#                                grav_ylim=grav_ylim, show_wells=show_wells,
#                                figsize=figsize)
#         frames.append(self._fig_to_pil(fig, dpi))
#         plt.close(fig)

#         # One frame per action
#         for i, action_idx in enumerate(actions):
#             obs, reward, terminated, truncated, info = self.step(action_idx)
#             infos.append(info)

#             fig = self._make_frame(step=i + 1, action_idx=action_idx,
#                                    grav_ylim=grav_ylim, show_wells=show_wells,
#                                    figsize=figsize)
#             frames.append(self._fig_to_pil(fig, dpi))
#             plt.close(fig)

#             if terminated:
#                 print(f"*** LEAKAGE at step {i + 1} — GIF ends here ***")
#                 break
#             if truncated:
#                 print(f"*** MAX STEPS reached at step {i + 1} ***")
#                 break

#         # Save GIF
#         if frames:
#             duration_ms = int(1000 / fps)
#             # Hold the last frame longer so the viewer can see the final state
#             durations = [duration_ms] * len(frames)
#             durations[-1] = duration_ms * 4

#             frames[0].save(
#                 save_path,
#                 save_all=True,
#                 append_images=frames[1:],
#                 duration=durations,
#                 loop=0,
#             )
#             print(f"GIF saved to: {save_path}  "
#                   f"({len(frames)} frames, {fps} fps)")

#         return infos

#     @staticmethod
#     def _fig_to_pil(fig, dpi: int = 100):
#         """Convert a matplotlib Figure to a PIL Image (for GIF creation)."""
#         from PIL import Image
#         import io
#         fig.set_dpi(dpi)
#         canvas = FigureCanvas(fig)
#         buf = io.BytesIO()
#         canvas.print_png(buf)
#         buf.seek(0)
#         return Image.open(buf).convert("RGB")

#     def get_action_meanings(self) -> List[str]:
#         """Return human-readable descriptions of each action."""
#         meanings = []
#         for i, (r1, r2, r3) in enumerate(DISCRETE_ACTIONS):
#             total = r1 + r2 + r3
#             f1, f2, f3 = r1/total, r2/total, r3/total
#             meanings.append(f"Action {i}: R1={f1:.1%}, R2={f2:.1%}, R3={f3:.1%}")
#         return meanings


# # Backwards compatibility alias
# SpillpointPOMDPEnv = MultiRegionCO2StorageEnv


# print("Multi-Region Environment (v2) code complete!")




# # ==============================================================================
# # ================================= NEW CODE ===================================
# # ==============================================================================
# def save_static_frames(env, actions, save_dir="saved_frames", dpi=150, show_wells=None):
#     """
#     Runs the environment through a list of actions and saves each step as a static PNG.
#     """
#     # Create the output directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)

#     # Pre-compute gravity y-limits to keep the graphs stable (optional but recommended)
#     env.reset()
#     grav_min, grav_max = 0.0, 0.0
#     for action_idx in actions:
#         env.step(action_idx)
#         _, grav_y = env.sim_env.get_time_lapse_gravity()
#         grav_min = min(grav_min, float(grav_y.min()))
#         grav_max = max(grav_max, float(grav_y.max()))
#     pad = max(abs(grav_max - grav_min) * 0.10, 0.5)
#     grav_ylim = (grav_min - pad, grav_max + pad)

#     # Reset environment to start the actual run
#     env.reset()

#     # Save Step 0 (Empty Reservoir)
#     fig = env._make_frame(step=0, action_idx=None, grav_ylim=grav_ylim, show_wells=show_wells)
#     fig.savefig(os.path.join(save_dir, "frame_00.png"), bbox_inches='tight', dpi=dpi)
#     plt.close(fig) # Close the figure to free up memory

#     # Loop through actions and save subsequent frames
#     for i, action_idx in enumerate(actions):
#         obs, reward, terminated, truncated, info = env.step(action_idx)

#         fig = env._make_frame(step=i + 1, action_idx=action_idx, grav_ylim=grav_ylim, show_wells=show_wells)
        
#         # Save the figure (e.g., frame_01.png, frame_02.png)
#         filename = os.path.join(save_dir, f"frame_{i + 1:02d}.png")
#         fig.savefig(filename, bbox_inches='tight', dpi=dpi)
#         plt.close(fig)

#         if terminated:
#             print(f"*** LEAKAGE at step {i + 1} — stopping early ***")
#             break
#         if truncated:
#             print(f"*** MAX STEPS reached at step {i + 1} ***")
#             break
            
#     print(f"Successfully saved images to the '{save_dir}' directory.")