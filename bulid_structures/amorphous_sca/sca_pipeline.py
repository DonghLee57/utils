#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCA pipeline (single file, no argparse):
- STEP 1) Build SCA YAML parameters from precomputed rdfadf outputs (prdf_*.out, cn_*.out).
- STEP 2) Run SCA seed-coordinate structure generation.
- STEP 3) Write output files (vasp + loop.extxyz + optional log).

User workflow
-------------
Run:
    python sca_pipeline.py

Then control what happens by editing the "USER SETTINGS" block in main().

Input files
-----------
A) Base YAML (base.yaml):
- Contains structure definition (lattice/density/chemical_formula or density/atom_types/num_atoms)
- Contains auto_params block specifying rdf_dir + pair_cutoffs to inject d_min/d_max/prob_*.

B) rdfadf outputs (already generated):
- prdf_*.out : two columns (r, g_AB(r))
- cn_*.out   : three columns (CN, count, probability)

Outputs
-------
- Generated YAML (input_generated.yaml) if enabled
- sca_out.vasp (or cfg['output'])
- loop.extxyz (if enabled)
- sca.log (if log enabled)
"""

# =============================================================================
# Utilities
# =============================================================================

import os
import re
import math
import yaml
import numpy as np

from ase import Atoms
from ase.io import write
from ase.data import atomic_masses, chemical_symbols, atomic_numbers

N_A = 6.02214076e23  # 1/mol
ANGSTROM_TO_CM = 1e-8


# ---------------------------
# Chemistry / cell utilities
# ---------------------------

def parse_chemical_formula(formula: str) -> dict:
    """
    Parse formula like 'Si1O2' -> {'Si': 1, 'O': 2}.
    """
    pattern = r"([A-Z][a-z]*)(\d*)"
    matches = re.findall(pattern, formula)
    out = {}
    for el, cnt in matches:
        out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
    return out


def process_lattice(lattice):
    """
    Convert lattice specification to a 3x3 cell matrix.

    - scalar: cubic cell (a=a0)
    - 3x3: direct
    """
    if isinstance(lattice, (float, int)):
        return np.eye(3) * float(lattice)
    arr = np.array(lattice, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError("lattice must be a scalar or a 3x3 matrix")
    return arr


def calculate_num_atoms_from_density(lattice: np.ndarray, density: float, chemical_formula: dict) -> (list, list):
    """
    Compute atom counts consistent with density for the provided cell volume. [file:6]
    """
    volume_a3 = abs(np.linalg.det(lattice))
    volume_cm3 = volume_a3 * (ANGSTROM_TO_CM ** 3)

    def amu(el):
        return atomic_masses[chemical_symbols.index(el)]  # g/mol

    molar_mass = sum(amu(el) * n for el, n in chemical_formula.items())  # g/mol
    target_formula_units = (density * volume_cm3 * N_A) / molar_mass
    fu = max(1, int(round(target_formula_units)))

    atom_types = list(chemical_formula.keys())
    num_atoms = [int(chemical_formula[el] * fu) for el in atom_types]
    return atom_types, num_atoms


def calculate_cubic_lattice_from_density(num_atoms: list, density: float, atom_types: list) -> np.ndarray:
    """
    Derive cubic lattice a from total mass and density. [file:6]
    """
    total_mass_g_per_mol = 0.0
    for el, n in zip(atom_types, num_atoms):
        total_mass_g_per_mol += atomic_masses[chemical_symbols.index(el)] * float(n)

    total_mass_g = total_mass_g_per_mol / N_A
    volume_cm3 = total_mass_g / float(density)
    volume_a3 = volume_cm3 / (ANGSTROM_TO_CM ** 3)
    a = volume_a3 ** (1.0 / 3.0)
    return np.eye(3) * a


def calculate_mass_density_g_cm3(atoms: Atoms) -> float:
    """
    Compute density in g/cm^3. [file:6]
    """
    masses_amu = atomic_masses[atoms.numbers]  # g/mol
    total_mass_g = float(np.sum(masses_amu) / N_A)
    volume_cm3 = float(atoms.get_volume() * (ANGSTROM_TO_CM ** 3))
    return total_mass_g / volume_cm3


def cell_as_str(cell: np.ndarray) -> str:
    """
    Compact pretty-print for a 3x3 cell.
    """
    return np.array2string(np.array(cell, dtype=float), precision=6, suppress_small=False)


# ---------------------------
# PBC distance utilities
# ---------------------------

def wrap_position(pos: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    """
    Wrap Cartesian position into the periodic cell.
    """
    frac = pos @ inv_cell
    frac = frac - np.floor(frac)
    return frac @ cell


def pbc_displacement(pos_i: np.ndarray, pos_j: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    """
    Minimum-image displacement under PBC, consistent with ASE mic=True used in rdfadf. [file:5]
    """
    d = pos_j - pos_i
    df = d @ inv_cell
    df -= np.round(df)
    return df @ cell


def pbc_distance(pos_i: np.ndarray, pos_j: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> float:
    """
    Minimum-image distance under PBC.
    """
    return float(np.linalg.norm(pbc_displacement(pos_i, pos_j, cell, inv_cell)))


# ---------------------------
# Sampling utilities
# ---------------------------

def normalize_prob_dict(prob_dict: dict, keys: list, mask: dict | None = None) -> np.ndarray | None:
    """
    Normalize prob_dict into a probability vector aligned with keys.
    """
    p = np.array([float(prob_dict.get(k, 0.0)) for k in keys], dtype=float)
    if mask is not None:
        m = np.array([bool(mask.get(k, True)) for k in keys], dtype=bool)
        p = p * m.astype(float)
    s = p.sum()
    if s <= 0.0:
        return None
    return p / s


def sample_truncated_gaussian(rng: np.random.Generator, mu: float, sigma: float, lo: float, hi: float, max_tries: int = 200) -> float:
    """
    Truncated Gaussian sampling by rejection.
    """
    if sigma <= 0.0:
        return float(np.clip(mu, lo, hi))
    for _ in range(int(max_tries)):
        x = float(rng.normal(mu, sigma))
        if lo <= x <= hi:
            return x
    x = float(rng.normal(mu, sigma))
    return float(np.clip(x, lo, hi))


# =============================================================================
# rdfadf-output-based parameter inference
# =============================================================================

def canonical_pair(a: str, b: str) -> tuple[str, str]:
    """
    Canonical ordering by atomic number; tie -> alphabetical.
    """
    za, zb = atomic_numbers[a], atomic_numbers[b]
    if za < zb:
        return a, b
    if za > zb:
        return b, a
    return (a, b) if a <= b else (b, a)


def parse_pair_key(s: str) -> tuple[str, str]:
    """
    Parse "A-B" -> (A,B).
    """
    if "-" not in s:
        raise ValueError(f"Pair key must look like 'A-B': {s}")
    a, b = s.split("-", 1)
    return a.strip(), b.strip()


def resolve_pair_file(directory: str, prefix: str, a: str, b: str, ext: str = ".out") -> str:
    """
    Resolve pair file path for A-B with symmetric fallback:
    - Try canonical(A,B) first (atomic-number order), then reversed.
    - Return the first existing file.

    This implements your "equivalent pair -> read one file automatically" request. [file:5]
    """
    A0, B0 = canonical_pair(a, b)
    cand = [
        os.path.join(directory, f"{prefix}_{A0}_{B0}{ext}"),
        os.path.join(directory, f"{prefix}_{B0}_{A0}{ext}"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing {prefix} file for pair {a}-{b} (tried: {cand})")


def load_two_column_xy(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load prdf file: columns (r, g(r)). [file:5]
    """
    arr = np.loadtxt(filename, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected >=2 columns: {filename}")
    return arr[:, 0], arr[:, 1]


def load_cn_distribution(filename: str) -> dict[int, float]:
    """
    Load CN distribution: columns (CN, count, probability). [file:5]
    Returns CN -> probability.
    """
    arr = np.loadtxt(filename, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns: {filename}")
    cn = arr[:, 0].astype(int)
    p = arr[:, 2].astype(float)
    s = p.sum()
    if s <= 0:
        raise ValueError(f"CN probability sum is zero: {filename}")
    p = p / s
    return {int(c): float(pi) for c, pi in zip(cn, p)}


def moving_average(y: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Simple moving average smoothing for PRDF.
    """
    window = int(max(1, window))
    if window == 1:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")


def estimate_dmin_from_prdf_threshold(
    r: np.ndarray,
    g: np.ndarray,
    threshold: float = 0.01,
    r_search_max: float | None = None,
) -> float:
    """
    Estimate d_min by threshold crossing on PRDF rising edge.
    The SCA paper defines d_min where g(r)=0.01. [file:1]
    """
    if len(r) != len(g) or len(r) < 5:
        raise ValueError("Invalid PRDF arrays")

    i1 = len(r) if (r_search_max is None) else int(np.searchsorted(r, r_search_max))
    r2 = r[:i1]
    g2 = g[:i1]
    if len(r2) < 5:
        raise ValueError("Not enough PRDF points in search window")

    idx = np.where(g2 >= threshold)[0]
    if len(idx) == 0:
        return float(r2[0])

    i = int(idx[0])
    if i == 0:
        return float(r2[0])

    x0, y0 = float(r2[i - 1]), float(g2[i - 1])
    x1, y1 = float(r2[i]), float(g2[i])
    if abs(y1 - y0) < 1e-12:
        return float(x1)
    t = (threshold - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def mean_cn_from_distribution(p_cn: dict[int, float]) -> float:
    """
    Mean CN = sum_N N*P(N).
    """
    return float(sum(int(n) * float(p) for n, p in p_cn.items()))


def convolve_cn_distributions(p_list: list[dict[int, float]], cn_max: int = 12) -> dict[int, float]:
    """
    Approximate total CN distribution by convolution of pairwise CN distributions. [file:5]
    """
    p_tot = {0: 1.0}
    for p in p_list:
        new = {}
        for n1, p1 in p_tot.items():
            for n2, p2 in p.items():
                n = int(n1) + int(n2)
                if n > cn_max:
                    continue
                new[n] = new.get(n, 0.0) + float(p1) * float(p2)
        s = sum(new.values())
        if s <= 0:
            raise ValueError("Convolution produced zero probability mass")
        p_tot = {n: v / s for n, v in new.items()}
    return p_tot


def build_sca_params_from_existing_prdf_cn(
    atom_types: list[str],
    rdf_dir: str,
    pair_cutoffs: dict[tuple[str, str], float],
    prdf_prefix: str = "prdf",
    cn_prefix: str = "cn",
    threshold: float = 0.01,
    smooth_window: int = 7,
    cn_max: int = 12,
) -> dict:
    """
    Build YAML-ready SCA parameters from existing prdf/cn files.

    Policy:
    - d_max(A-B) = user-provided cutoff for canonical(A,B). [file:5]
    - d_min(A-B) from PRDF threshold (paper: g=0.01), search up to cutoff for stability. [file:1]
    - A-B / B-A equivalent: file resolver chooses one existing file; d_min/d_max are written for all ordered pairs.

    Returns
    -------
    dict with keys: d_min, d_max, prob_types, prob_cn
    """
    # canonicalize cutoffs
    cutoff_map = {}
    for (a, b), c in pair_cutoffs.items():
        A0, B0 = canonical_pair(a, b)
        cutoff_map[(A0, B0)] = float(c)

    d_min = {}
    d_max = {}
    prob_types = {A: {} for A in atom_types}
    prob_cn = {A: {} for A in atom_types}

    # d_min/d_max
    for A in atom_types:
        for B in atom_types:
            A0, B0 = canonical_pair(A, B)
            if (A0, B0) not in cutoff_map:
                raise ValueError(f"Missing cutoff for canonical pair {(A0, B0)}")
            cutoff = cutoff_map[(A0, B0)]

            prdf_path = resolve_pair_file(rdf_dir, prdf_prefix, A, B)
            r, g = load_two_column_xy(prdf_path)
            g_s = moving_average(g, window=smooth_window)

            dmin = estimate_dmin_from_prdf_threshold(r, g_s, threshold=threshold, r_search_max=cutoff)

            d_min[f"{A}-{B}"] = float(min(dmin, cutoff))
            d_max[f"{A}-{B}"] = float(cutoff)

    # prob_types from mean pair CN fractions
    mean_pair_cn = {A: {B: 0.0 for B in atom_types} for A in atom_types}
    for A in atom_types:
        for B in atom_types:
            cn_path = resolve_pair_file(rdf_dir, cn_prefix, A, B)
            p_ab = load_cn_distribution(cn_path)
            mean_pair_cn[A][B] = mean_cn_from_distribution(p_ab)

        denom = sum(mean_pair_cn[A].values())
        if denom <= 0:
            for B in atom_types:
                prob_types[A][B] = 1.0 / len(atom_types)
        else:
            for B in atom_types:
                prob_types[A][B] = float(mean_pair_cn[A][B] / denom)

    # prob_cn by convolution approximation
    for A in atom_types:
        plist = []
        for B in atom_types:
            cn_path = resolve_pair_file(rdf_dir, cn_prefix, A, B)
            plist.append(load_cn_distribution(cn_path))
        p_total = convolve_cn_distributions(plist, cn_max=cn_max)
        prob_cn[A] = {int(n): float(p) for n, p in sorted(p_total.items())}

    return {"d_min": d_min, "d_max": d_max, "prob_types": prob_types, "prob_cn": prob_cn}


# =============================================================================
# YAML config workflow
# =============================================================================

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_base_yaml_and_finalize(cfg: dict) -> dict:
    """
    Normalize structure-defining config:
    - determine lattice/atom_types/num_atoms, set defaults, create _expected. [file:6]
    """
    cfg = dict(cfg)

    if ("lattice" in cfg) and ("density" in cfg) and ("chemical_formula" in cfg):
        cfg["lattice"] = process_lattice(cfg["lattice"])
        atom_types, num_atoms = calculate_num_atoms_from_density(
            cfg["lattice"], float(cfg["density"]), parse_chemical_formula(cfg["chemical_formula"])
        )
        cfg["atom_types"] = atom_types
        cfg["num_atoms"] = num_atoms

    elif ("density" in cfg) and ("num_atoms" in cfg) and ("atom_types" in cfg) and ("lattice" not in cfg):
        cfg["lattice"] = calculate_cubic_lattice_from_density(cfg["num_atoms"], float(cfg["density"]), cfg["atom_types"])

    elif ("lattice" in cfg) and ("num_atoms" in cfg) and ("atom_types" in cfg):
        cfg["lattice"] = process_lattice(cfg["lattice"])

    else:
        raise ValueError(
            "Invalid base YAML. Provide (lattice,density,chemical_formula) or (density,num_atoms,atom_types) or (lattice,num_atoms,atom_types)."
        )

    # Defaults (from sca.txt style) [file:6]
    cfg.setdefault("log", 0)
    cfg.setdefault("rng_seed", 0)
    cfg.setdefault("output", "sca_out.vasp")
    cfg.setdefault("save_loop", True)
    cfg.setdefault("loop_file", "loop.extxyz")
    cfg.setdefault("loop_stride", 1)
    cfg.setdefault("max_trials", 4000)
    cfg.setdefault("max_trials_per_type", 1200)
    cfg.setdefault("allow_extra_bonds_to_not_full", True)

    # Gaussian bond-length sampling defaults
    cfg.setdefault("bond_length_sampling", "gaussian")
    cfg.setdefault("gaussian_sigma_scale", 6.0)
    cfg.setdefault("gaussian_max_tries", 200)

    # Expected for final report
    cfg["_expected"] = {}
    cfg["_expected"]["density_g_cm3"] = float(cfg.get("density", float("nan")))
    cfg["_expected"]["lattice"] = np.array(cfg["lattice"], dtype=float)
    cfg["_expected"]["counts_by_type"] = {t: int(n) for t, n in zip(cfg["atom_types"], cfg["num_atoms"])}
    cfg["_expected"]["total_atoms"] = int(sum(cfg["num_atoms"]))

    return cfg


def inject_auto_params_if_enabled(cfg: dict) -> dict:
    """
    If cfg['auto_params']['enabled'] is True, build d_min/d_max/prob_* from rdf outputs and inject. [file:5]
    """
    ap = cfg.get("auto_params", {})
    if not ap or not bool(ap.get("enabled", False)):
        return cfg

    rdf_dir = str(ap.get("rdf_dir", "."))
    prdf_prefix = str(ap.get("prdf_prefix", "prdf"))
    cn_prefix = str(ap.get("cn_prefix", "cn"))
    threshold = float(ap.get("threshold", 0.01))
    smooth_window = int(ap.get("smooth_window", 7))
    cn_max = int(ap.get("cn_max", 12))

    raw = ap.get("pair_cutoffs", {})
    if not raw:
        raise ValueError("auto_params.enabled is true but pair_cutoffs is missing/empty")

    pair_cutoffs = {}
    for k, v in raw.items():
        a, b = parse_pair_key(k)
        A0, B0 = canonical_pair(a, b)
        pair_cutoffs[(A0, B0)] = float(v)

    params = build_sca_params_from_existing_prdf_cn(
        atom_types=list(cfg["atom_types"]),
        rdf_dir=rdf_dir,
        pair_cutoffs=pair_cutoffs,
        prdf_prefix=prdf_prefix,
        cn_prefix=cn_prefix,
        threshold=threshold,
        smooth_window=smooth_window,
        cn_max=cn_max,
    )

    cfg.update(params)
    return cfg


def ensure_full_sca_fields(cfg: dict):
    """
    Ensure d_min/d_max/prob_types/prob_cn exist and are complete. [file:6]
    """
    for k in ["d_min", "d_max", "prob_types", "prob_cn"]:
        if k not in cfg:
            raise ValueError(f"Missing required SCA field: {k}")

    at = list(cfg["atom_types"])
    for a in at:
        for b in at:
            k = f"{a}-{b}"
            ks = f"{b}-{a}"
            if k not in cfg["d_min"] and ks in cfg["d_min"]:
                cfg["d_min"][k] = cfg["d_min"][ks]
            if k not in cfg["d_max"] and ks in cfg["d_max"]:
                cfg["d_max"][k] = cfg["d_max"][ks]
            if k not in cfg["d_min"] or k not in cfg["d_max"]:
                raise ValueError(f"Missing d_min/d_max for pair {k}")

    for c in at:
        cfg["prob_types"].setdefault(c, {})
        for n in at:
            cfg["prob_types"][c].setdefault(n, 0.0)

    for c in at:
        if c not in cfg["prob_cn"]:
            raise ValueError(f"Missing prob_cn for type {c}")


def build_full_config_from_base_yaml(base_yaml_path: str) -> dict:
    """
    High-level helper:
    read base.yaml -> finalize structure -> inject auto params -> validate.
    """
    cfg0 = read_yaml(base_yaml_path)
    cfg = load_base_yaml_and_finalize(cfg0)
    cfg = inject_auto_params_if_enabled(cfg)
    ensure_full_sca_fields(cfg)
    return cfg


def write_outputs(cfg: dict, atoms: Atoms):
    """
    Write final structure output(s).
    """
    out = str(cfg.get("output", "sca_out.vasp"))
    write(out, atoms, format="vasp")
    print(f"Wrote: {out} (N={len(atoms)})")


# =============================================================================
# SCA generator (class region)
# =============================================================================

class SCAGenerator:
    """
    SCA seed-coordinate generator (based on your sca.txt structure). [file:6]

    Differences from the original:
    - Bond length sampling in [d_min, d_max] supports Gaussian (default) rather than uniform. [file:6]
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cell = np.array(cfg["lattice"], dtype=float)
        self.inv_cell = np.linalg.inv(self.cell)

        self.atom_types = list(cfg["atom_types"])
        self.num_atoms_target = list(cfg["num_atoms"])

        self.d_min = dict(cfg["d_min"])
        self.d_max = dict(cfg["d_max"])
        self.prob_types = cfg["prob_types"]
        self.prob_cn = cfg["prob_cn"]

        self.log = int(cfg.get("log", 0))
        self.rng = np.random.default_rng(int(cfg.get("rng_seed", 0)))

        self.max_trials = int(cfg.get("max_trials", 4000))
        self.max_trials_per_type = int(cfg.get("max_trials_per_type", 1200))
        self.allow_extra_bonds_to_not_full = bool(cfg.get("allow_extra_bonds_to_not_full", True))

        # Gaussian bond-length sampling
        self.bond_length_sampling = str(cfg.get("bond_length_sampling", "gaussian")).lower()
        self.gaussian_sigma_scale = float(cfg.get("gaussian_sigma_scale", 6.0))
        self.gaussian_max_tries = int(cfg.get("gaussian_max_tries", 200))

        self.atoms = Atoms(cell=self.cell, pbc=True)

        self.types = []
        self.cn_target = []
        self.cn_current = []
        self.neigh = []

        self.remaining = {t: int(n) for t, n in zip(self.atom_types, self.num_atoms_target)}
        self.total_target = int(sum(self.num_atoms_target))

        # Logging [file:6]
        self.logfp = open("sca.log", "w", encoding="utf-8") if self.log else None
        if self.logfp is not None:
            self.logfp.write("# event index type cn0 x y z\n")
            self.logfp.flush()

        # Loop trajectory output [file:6]
        self.save_loop = bool(cfg.get("save_loop", True))
        self.loop_file = str(cfg.get("loop_file", "loop.extxyz"))
        self.loop_stride = int(cfg.get("loop_stride", 1))
        self._n_events = 0
        if self.save_loop:
            open(self.loop_file, "w", encoding="utf-8").close()

    def close(self):
        if self.logfp is not None:
            self.logfp.close()
            self.logfp = None

    def _key(self, a, b):
        return f"{a}-{b}"

    def _dmin(self, a, b):
        return float(self.d_min[self._key(a, b)])

    def _dmax(self, a, b):
        return float(self.d_max[self._key(a, b)])

    def is_fully(self, idx: int) -> bool:
        return self.cn_current[idx] >= self.cn_target[idx]

    def sample_cn0(self, atom_type: str) -> int:
        cn_map = self.prob_cn[atom_type]
        cn_values = [int(k) for k in cn_map.keys()]
        probs = np.array([float(cn_map.get(str(k), cn_map.get(k, 0.0))) for k in cn_values], dtype=float)
        probs = probs / probs.sum()
        return int(self.rng.choice(cn_values, p=probs))

    def sample_type_with_stoichiometry(self) -> str | None:
        types = [t for t in self.atom_types if self.remaining[t] > 0]
        if not types:
            return None
        w = np.array([self.remaining[t] for t in types], dtype=float)
        w /= w.sum()
        return str(self.rng.choice(types, p=w))

    def sorted_neighbor_candidates(self, center_type: str) -> list[str]:
        cand = []
        for t in self.atom_types:
            if self.remaining[t] <= 0:
                continue
            prob = float(self.prob_types.get(center_type, {}).get(t, 0.0))
            if prob <= 0.0:
                continue
            cand.append((prob * self.remaining[t], prob, t))
        cand.sort(reverse=True)
        return [t for _, _, t in cand]

    def append_atom_no_record(self, atom_type: str, pos: np.ndarray, cn0: int):
        self.atoms.append(atom_type)
        self.atoms.positions[-1] = pos
        self.types.append(atom_type)
        self.cn_target.append(int(cn0))
        self.cn_current.append(0)
        self.neigh.append(set())

    def add_bond(self, i: int, j: int):
        if j in self.neigh[i]:
            return
        self.neigh[i].add(j)
        self.neigh[j].add(i)
        self.cn_current[i] += 1
        self.cn_current[j] += 1

    def record_event(self, event: str, atom_index: int):
        self._n_events += 1

        if self.logfp is not None and len(self.atoms) > 0:
            pos = self.atoms.positions[atom_index]
            t = self.types[atom_index]
            cn0 = self.cn_target[atom_index]
            self.logfp.write(f"{event} {atom_index} {t} {cn0} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")
            self.logfp.flush()

        if self.save_loop:
            if self.loop_stride <= 1 or (self._n_events % self.loop_stride == 0):
                self.atoms.set_array("cn_target", np.array(self.cn_target, dtype=int))
                self.atoms.set_array("cn_current", np.array(self.cn_current, dtype=int))
                self.atoms.info["event"] = str(event)
                self.atoms.info["event_index"] = int(self._n_events)
                write(self.loop_file, self.atoms, format="extxyz", append=True)

    def check_seed_position(self, pos: np.ndarray, atom_type: str) -> bool:
        """
        Seed condition: no bonds upon insertion => distance >= d_max to all existing atoms. [file:1]
        """
        n = len(self.types)
        if n == 0:
            return True
        for k in range(n):
            tk = self.types[k]
            dist = pbc_distance(pos, self.atoms.positions[k], self.cell, self.inv_cell)
            if dist < self._dmax(atom_type, tk):
                return False
            if dist < self._dmin(atom_type, tk):
                return False
        return True

    def sample_bond_length(self, rmin: float, rmax: float) -> float:
        if rmax <= rmin:
            return float(rmin)
        if self.bond_length_sampling == "uniform":
            return float(self.rng.uniform(rmin, rmax))
        mu = 0.5 * (rmin + rmax)
        sigma = (rmax - rmin) / float(self.gaussian_sigma_scale)
        return sample_truncated_gaussian(self.rng, mu, sigma, rmin, rmax, max_tries=self.gaussian_max_tries)

    def propose_shell_point(self, center_pos: np.ndarray, rmin: float, rmax: float) -> np.ndarray:
        v = self.rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = self.sample_bond_length(rmin, rmax)
        pos = center_pos + v * r
        return wrap_position(pos, self.cell, self.inv_cell)

    def validate_and_collect_extra_bonds(self, seed_i: int, cand_pos: np.ndarray, cand_type: str, cand_cn0: int):
        seed_type = self.types[seed_i]
        d = pbc_distance(self.atoms.positions[seed_i], cand_pos, self.cell, self.inv_cell)
        if not (self._dmin(seed_type, cand_type) <= d < self._dmax(seed_type, cand_type)):
            return None

        extra_bonds = []
        n = len(self.types)

        for k in range(n):
            if k == seed_i:
                continue
            tk = self.types[k]
            dist = pbc_distance(self.atoms.positions[k], cand_pos, self.cell, self.inv_cell)

            if dist < self._dmin(cand_type, tk):
                return None

            if dist < self._dmax(cand_type, tk):
                if self.is_fully(k):
                    return None
                if not self.allow_extra_bonds_to_not_full:
                    return None
                if self.cn_current[k] + 1 > self.cn_target[k]:
                    return None
                extra_bonds.append(k)

        if 1 + len(extra_bonds) > int(cand_cn0):
            return None

        return extra_bonds

    def place_new_seed(self) -> int | None:
        t = self.sample_type_with_stoichiometry()
        if t is None:
            return None
        cn0 = self.sample_cn0(t)

        for _ in range(self.max_trials):
            if len(self.types) == 0:
                pos = np.zeros(3)
            else:
                pos = (self.rng.random(3) @ self.cell)
            pos = wrap_position(pos, self.cell, self.inv_cell)

            if self.check_seed_position(pos, t):
                self.append_atom_no_record(t, pos, cn0)
                self.remaining[t] -= 1
                idx = len(self.types) - 1
                self.record_event("seed", idx)
                return idx
        return None

    def fill_coordination_for_atom(self, i: int, active_queue: list[int]) -> bool:
        if self.is_fully(i):
            return True

        while (not self.is_fully(i)) and (sum(self.remaining.values()) > 0):
            center_type = self.types[i]
            cand_types = self.sorted_neighbor_candidates(center_type)
            if not cand_types:
                return False

            placed_one = False
            for nt in cand_types:
                if self.remaining[nt] <= 0:
                    continue

                cn0_j = self.sample_cn0(nt)
                rmin = self._dmin(center_type, nt)
                rmax = self._dmax(center_type, nt)

                for _ in range(self.max_trials_per_type):
                    cand_pos = self.propose_shell_point(self.atoms.positions[i], rmin, rmax)
                    extra = self.validate_and_collect_extra_bonds(i, cand_pos, nt, cn0_j)
                    if extra is None:
                        continue

                    self.append_atom_no_record(nt, cand_pos, cn0_j)
                    j = len(self.types) - 1
                    self.remaining[nt] -= 1

                    self.add_bond(i, j)
                    for k in extra:
                        self.add_bond(k, j)

                    self.record_event("coord", j)

                    if not self.is_fully(j):
                        active_queue.append(j)

                    placed_one = True
                    break

                if placed_one:
                    break

            if not placed_one:
                return False

        return True

    def run(self) -> Atoms:
        active = []
        stall_counter = 0

        while len(self.types) < self.total_target:
            if not active:
                seed_idx = self.place_new_seed()
                if seed_idx is None:
                    stall_counter += 1
                    if stall_counter > 50:
                        raise RuntimeError("Failed to place seeds repeatedly. Check d_min/d_max, density, lattice.")
                    continue
                active.append(seed_idx)

            i = active.pop(0)
            ok = self.fill_coordination_for_atom(i, active)
            if not ok:
                active.append(i)
                stall_counter += 1
                if stall_counter % 10 == 0:
                    self.place_new_seed()
            else:
                stall_counter = max(0, stall_counter - 1)

        self.atoms.wrap()
        order = np.argsort(self.atoms.numbers)
        self.atoms = self.atoms[order]

        if len(self.atoms) > 0:
            self.record_event("final", 0)

        return self.atoms

    def final_report(self):
        actual_counts = {t: 0 for t in self.atom_types}
        for s in self.atoms.get_chemical_symbols():
            actual_counts[s] = actual_counts.get(s, 0) + 1

        exp = self.cfg.get("_expected", {})
        expected_counts = exp.get("counts_by_type", {t: None for t in self.atom_types})
        expected_total = exp.get("total_atoms", None)

        actual_density = calculate_mass_density_g_cm3(self.atoms)
        target_density = exp.get("density_g_cm3", float("nan"))

        actual_cell = np.array(self.atoms.cell.array, dtype=float)
        target_cell = np.array(exp.get("lattice", actual_cell), dtype=float)
        cell_diff = actual_cell - target_cell

        lines = []
        lines.append("===== SCA final report =====")
        lines.append(f"Total atoms: actual={len(self.atoms)}, expected={expected_total}")
        lines.append("Counts by type (actual vs expected, diff):")
        for t in self.atom_types:
            a = actual_counts.get(t, 0)
            e = expected_counts.get(t, None)
            if e is None:
                lines.append(f"  - {t}: actual={a}, expected=NA")
            else:
                lines.append(f"  - {t}: actual={a}, expected={e}, diff={a - e}")

        if not (math.isnan(target_density) or math.isinf(target_density)):
            lines.append(f"Density (g/cm3): actual={actual_density:.6f}, target={target_density:.6f}, diff={actual_density - target_density:+.6f}")
        else:
            lines.append(f"Density (g/cm3): actual={actual_density:.6f}, target=NA")

        lines.append("Cell (Angstrom) compare:")
        lines.append(f"  target cell:\n{cell_as_str(target_cell)}")
        lines.append(f"  actual cell:\n{cell_as_str(actual_cell)}")
        lines.append(f"  cell diff Fro-norm: {np.linalg.norm(cell_diff):.6e}")

        lines.append("Bond-length sampling:")
        lines.append(f"  mode={self.bond_length_sampling}, gaussian_sigma_scale={self.gaussian_sigma_scale}, gaussian_max_tries={self.gaussian_max_tries}")

        report = "\n".join(lines)
        print(report)
        if self.logfp is not None:
            self.logfp.write("\n" + report + "\n")
            self.logfp.flush()


# =============================================================================
# main(): user-controlled steps
# =============================================================================

def main():
    """
    Main routine without argparse:
    - Edit USER SETTINGS below and run `python sca_pipeline.py`.
    """
    # -------------------------
    # USER SETTINGS (edit this)
    # -------------------------
    BASE_YAML = "input.yaml"
    GENERATED_YAML = "input_generated.yaml"

    DO_MAKE_INPUT = False   # Step 1: build full YAML (inject d_min/d_max/prob_*)
    DO_RUN_SCA = True       # Step 2: run SCA and write structure outputs

    # -------------------------
    # STEP 1) Make input YAML
    # -------------------------
    if DO_MAKE_INPUT:
        cfg = build_full_config_from_base_yaml(BASE_YAML)
        write_yaml(GENERATED_YAML, cfg)
        print(f"Wrote generated YAML: {GENERATED_YAML}")
    else:
        # If skipping input generation, still need a config to run SCA (from GENERATED_YAML or BASE_YAML).
        cfg = None

    # -------------------------
    # STEP 2) Run SCA
    # -------------------------
    if DO_RUN_SCA:
        # Prefer generated yaml if it exists and Step 1 ran; otherwise run from base.yaml (must already have full fields).
        yaml_to_run = GENERATED_YAML if (DO_MAKE_INPUT and os.path.exists(GENERATED_YAML)) else BASE_YAML
        cfg_run = build_full_config_from_base_yaml(yaml_to_run)  # also injects auto_params if enabled
        gen = SCAGenerator(cfg_run)
        try:
            atoms = gen.run()
            gen.final_report()
            write_outputs(cfg_run, atoms)
            if bool(cfg_run.get("save_loop", True)):
                print(f"Wrote: {cfg_run.get('loop_file','loop.extxyz')} (multi-frame trajectory)")
        finally:
            gen.close()

if __name__ == "__main__":
    main()
