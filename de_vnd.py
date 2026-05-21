"""
DE-VND – Evolución Diferencial Híbrida con VND para NWJSSP
============================================================
Combina Evolución Diferencial (DE) como motor de exploración global con
Variable Neighborhood Descent (VND) como refinamiento local.

Estrategia híbrida (Memetic / Evolutionary Local Search):
  1. Población inicial de NS individuos (vectores de claves reales).
  2. En cada generación, para cada individuo j:
       a. Mutación DE/rand/1:  v = P_c + F*(P_a - P_b)
       b. Cruce binomial:      u = crossover(P_j, v, CR)
       c. Evaluación:          f(u)
       d. Selección greedy:    si f(u) < f(P_j) → P_j ← u
  3. Cada `vnd_freq` generaciones, VND refina al mejor individuo.
  4. Se continúa hasta agotar el tiempo límite.

Control de tiempo — TODAS las fases respetan el deadline global:
  - Inicialización: verifica deadline antes de evaluar cada individuo;
    si el tiempo se agota, sale con la mejor solución parcial disponible.
  - Bucle DE: verifica deadline antes de cada individuo j dentro de la
    generación.
  - Fase VND: recibe como deadline el mínimo entre el deadline global y
    un presupuesto local (15% del tiempo total). Los vecindarios internos
    (2-opt, Swap, Insertion) de vnd.py verifican el deadline en cada
    iteración de su propio bucle interno.
  - Salida: en cualquier punto de corte se devuelve la mejor solución
    encontrada hasta ese momento, nunca None.

Parámetros:
    NS            : tamaño de la población                     (default 15)
    F             : factor de escala de mutación ∈ (0, 2]     (default 0.8)
    CR            : tasa de cruce ∈ [0, 1]                    (default 0.9)
    vnd_freq      : cada cuántas generaciones se aplica VND    (default 5)
    vnd_max_range : max_range para los vecindarios VND         (default 10)
    vnd_improve_n1/n2/n3: estrategia BI/FI por vecindario VND
    time_limit    : segundos máximos totales                   (default 3600)
    seed          : semilla aleatoria                          (default 42)
"""

import time
import numpy as np

from constructive import ConstructiveAlgorithm
from read_instances import calculate_flow_time
from vnd import evaluate_sequence, _NEIGHBORHOOD_FN


# ---------------------------------------------------------------------------
# Utilidades de codificación
# ---------------------------------------------------------------------------

def _keys_to_sequence(keys):
    return list(np.argsort(keys))


def _evaluate_keys(keys, algo):
    seq = _keys_to_sequence(keys)
    flow, starts = evaluate_sequence(seq, algo)
    return flow, starts


def _starts_to_keys(starts, n):
    """Convierte tiempos de inicio a vector de claves normalizadas."""
    order = sorted(range(n), key=lambda j: starts[j])
    keys = np.zeros(n)
    for rank, job in enumerate(order):
        keys[job] = rank / n
    return keys


# ---------------------------------------------------------------------------
# VND local con deadline global propagado
# ---------------------------------------------------------------------------

def _run_vnd_local(seq, algo, max_range,
                   improve_n1, improve_n2, improve_n3,
                   deadline):
    """
    Aplica VND estándar (3 vecindarios) sobre `seq`.

    El parámetro `deadline` es el límite de tiempo absoluto (time.time()).
    Se verifica:
      - Antes de entrar a cada vecindario j.
      - Dentro de cada vecindario (los bucles de vnd.py aceptan deadline).
    Devuelve (seq_mejorada, flow_time, starts).
    """
    strategies = [improve_n1.upper(), improve_n2.upper(), improve_n3.upper()]
    current_flow, current_starts = evaluate_sequence(seq, algo)

    j = 1
    while j <= 3:
        # Verificar tiempo ANTES de entrar a cada vecindario
        if time.time() >= deadline:
            break
        fn = _NEIGHBORHOOD_FN[(j, strategies[j - 1])]
        new_seq, new_flow, new_starts, improved = fn(
            seq, algo, max_range, deadline   # deadline global, no LOCAL_TIME_LIMIT
        )
        if improved:
            seq = new_seq
            current_flow = new_flow
            current_starts = new_starts
            j = 1
        else:
            j += 1

    return seq, current_flow, current_starts


# ---------------------------------------------------------------------------
# Clase principal DEVNDSearch
# ---------------------------------------------------------------------------

class DEVNDSearch:
    """
    DE híbrido con VND para NWJSSP.

    Parámetros
    ----------
    NS            : int   tamaño de población                     (default 15)
    F             : float factor de escala ∈ (0,2]               (default 0.8)
    CR            : float tasa de cruce ∈ [0,1]                  (default 0.9)
    vnd_freq      : int   frecuencia de VND (cada N generaciones) (default 5)
    vnd_max_range : int   max_range para VND                      (default 10)
    vnd_improve_n1: str   "BI"/"FI" para 2-opt                   (default "BI")
    vnd_improve_n2: str   "BI"/"FI" para Swap                    (default "BI")
    vnd_improve_n3: str   "BI"/"FI" para Insertion               (default "FI")
    time_limit    : float segundos máximos totales                (default 3600)
    seed          : int   semilla aleatoria                       (default 42)
    """

    def __init__(self, n, m, operations, release_dates,
                 NS: int = 15,
                 F: float = 0.8,
                 CR: float = 0.9,
                 vnd_freq: int = 5,
                 vnd_max_range: int = 10,
                 vnd_improve_n1: str = "BI",
                 vnd_improve_n2: str = "BI",
                 vnd_improve_n3: str = "FI",
                 time_limit: float = 3600.0,
                 seed: int = 42):
        self.n = n
        self.m = m
        self.operations = operations
        self.release_dates = release_dates
        self.NS = NS
        self.F = F
        self.CR = CR
        self.vnd_freq = vnd_freq
        self.vnd_max_range = vnd_max_range
        self.vnd_improve_n1 = vnd_improve_n1.upper()
        self.vnd_improve_n2 = vnd_improve_n2.upper()
        self.vnd_improve_n3 = vnd_improve_n3.upper()
        self.time_limit = time_limit
        self.seed = seed
        self._algo = ConstructiveAlgorithm(n, m, operations, release_dates)

        if not (0 < F <= 2):
            raise ValueError(f"F debe estar en (0, 2], recibido: {F}")
        if not (0 <= CR <= 1):
            raise ValueError(f"CR debe estar en [0, 1], recibido: {CR}")
        if NS < 4:
            raise ValueError("NS debe ser >= 4")

    # ------------------------------------------------------------------
    # Inicialización de población con control de tiempo
    # ------------------------------------------------------------------

    def _init_population(self, initial_solution, deadline):
        """
        Genera y evalúa la población inicial.
        Verifica el deadline antes de evaluar cada individuo.
        Retorna: (pop, fitness, starts_list, seq_list,
                  best_flow, best_starts, best_seq, n_evals)
        """
        rng = np.random.default_rng(self.seed)
        pop = rng.random((self.NS, self.n))

        if initial_solution is not None:
            pop[0] = _starts_to_keys(initial_solution, self.n)

        fitness = np.full(self.NS, np.inf)
        starts_list = [None] * self.NS
        seq_list = [None] * self.NS
        n_evals = 0

        best_flow = np.inf
        best_starts = None
        best_seq = None

        for i in range(self.NS):
            # Verificar tiempo ANTES de cada evaluación
            if time.time() >= deadline:
                break
            f, s = _evaluate_keys(pop[i], self._algo)
            fitness[i] = f
            starts_list[i] = s
            seq_list[i] = _keys_to_sequence(pop[i])
            n_evals += 1
            if f < best_flow:
                best_flow = f
                best_starts = s
                best_seq = seq_list[i]

        # Fallback: si el tiempo se agotó antes de evaluar cualquier individuo
        if best_starts is None and initial_solution is not None:
            best_starts = initial_solution
            best_flow, _ = calculate_flow_time(
                initial_solution, self.operations, self.release_dates
            )
            best_seq = sorted(range(self.n), key=lambda j: initial_solution[j])

        return pop, fitness, starts_list, seq_list, best_flow, best_starts, best_seq, n_evals

    # ------------------------------------------------------------------
    # Operadores DE
    # ------------------------------------------------------------------

    def _mutate(self, pop, j, rng):
        candidates = [i for i in range(self.NS) if i != j]
        a, b, c = rng.choice(candidates, size=3, replace=False)
        return pop[c] + self.F * (pop[a] - pop[b])

    def _crossover(self, target, mutant, rng):
        D = self.n
        r = int(rng.integers(0, D))
        mask = rng.random(D) <= self.CR
        mask[r] = True
        return np.where(mask, mutant, target)

    @staticmethod
    def _clip(v):
        return np.clip(v, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Presupuesto de tiempo para VND
    # ------------------------------------------------------------------

    def _vnd_deadline(self, global_deadline):
        """
        Calcula el deadline para una llamada VND:
        mínimo entre el deadline global y un presupuesto local de 15%
        del tiempo total, con un mínimo de 1 s para que valga la pena.
        Devuelve None si no queda tiempo suficiente.
        """
        remaining = global_deadline - time.time()
        if remaining <= 1.0:
            return None
        budget = min(remaining, self.time_limit * 0.15)
        return time.time() + budget

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def solve(self, initial_solution=None):
        """
        Ejecuta DE-VND respetando el time_limit en TODAS las fases.
        En cualquier punto de corte se devuelve la mejor solución encontrada.

        Returns
        -------
        job_start_times  : list[int]
        flow_time        : float
        computation_time : float   (milisegundos)
        n_evaluations    : int
        """
        start_t = time.time()
        deadline = start_t + self.time_limit
        rng = np.random.default_rng(self.seed)

        # ── Fase 1: Población inicial (con control de tiempo) ──────────
        (pop, fitness, starts_list, seq_list,
         best_flow, best_starts, best_seq, n_evaluations) = \
            self._init_population(initial_solution, deadline)

        # Si el tiempo se agotó durante la inicialización, retornar ya.
        if time.time() >= deadline:
            computation_time = (time.time() - start_t) * 1000
            return best_starts, best_flow, computation_time, n_evaluations

        best_idx = int(np.argmin(fitness))

        # ── Fase 2: Bucle de generaciones ─────────────────────────────
        generation = 0
        while time.time() < deadline:
            generation += 1

            for j in range(self.NS):
                # Verificar tiempo ANTES de cada individuo
                if time.time() >= deadline:
                    break

                # Mutación + cruce
                mutant = self._clip(self._mutate(pop, j, rng))
                trial = self._crossover(pop[j], mutant, rng)

                # Evaluación
                trial_flow, trial_starts = _evaluate_keys(trial, self._algo)
                n_evaluations += 1

                # Selección
                if trial_flow < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_flow
                    starts_list[j] = trial_starts
                    seq_list[j] = _keys_to_sequence(trial)

                    if trial_flow < best_flow:
                        best_flow = trial_flow
                        best_starts = trial_starts
                        best_seq = seq_list[j]
                        best_idx = j

            # ── Fase VND: intensificación local sobre el mejor ─────────
            if generation % self.vnd_freq == 0:
                vnd_dl = self._vnd_deadline(deadline)
                if vnd_dl is None:
                    # Sin tiempo suficiente para VND, salir del bucle principal
                    break

                new_seq, new_flow, new_starts = _run_vnd_local(
                    best_seq, self._algo,
                    self.vnd_max_range,
                    self.vnd_improve_n1,
                    self.vnd_improve_n2,
                    self.vnd_improve_n3,
                    vnd_dl,        # deadline = min(global, presupuesto local)
                )
                n_evaluations += 1

                if new_flow < best_flow:
                    best_flow = new_flow
                    best_starts = new_starts
                    best_seq = new_seq
                    # Reinsertar el individuo mejorado reemplazando al peor
                    worst_idx = int(np.argmax(fitness))
                    new_keys = _starts_to_keys(new_starts, self.n)
                    pop[worst_idx] = new_keys
                    fitness[worst_idx] = new_flow
                    starts_list[worst_idx] = new_starts
                    seq_list[worst_idx] = new_seq
                    best_idx = worst_idx

        computation_time = (time.time() - start_t) * 1000
        return best_starts, best_flow, computation_time, n_evaluations
