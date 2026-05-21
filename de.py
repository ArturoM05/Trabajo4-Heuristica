"""
DE – Evolución Diferencial para NWJSSP
========================================
Implementa el pseudocódigo de Evolución Diferencial (Storn & Price, 1997)
adaptado a espacios de permutaciones (secuencias de trabajos).

El DE clásico opera sobre vectores reales continuos. Para adaptarlo a
secuencias de trabajos (espacio discreto de permutaciones), se usa la
representación de claves aleatorias (random-key encoding):
  - Cada individuo P_j es un vector de D = n valores reales en [0, 1].
  - La secuencia de trabajos se obtiene ordenando los índices por sus valores:
      seq = argsort(P_j)  → orden ascendente de claves define prioridades.
  - La mutación DE opera sobre los vectores reales y produce v_j.
  - El cruce (binomial) mezcla v_j con P_j para generar u_j.
  - u_j se convierte a secuencia → se evalúa → selección greedy.

Control de tiempo:
  El deadline (start + time_limit) se propaga a TODAS las fases:
    - Inicialización de la población: se detiene si se agota el tiempo,
      devolviendo la mejor solución parcial disponible.
    - Bucle de generaciones: verifica el deadline antes de cada individuo.
    - Evaluación de vecinos individuales: operación atómica, no interrumpible,
      pero es O(n*m) y no puede exceder el límite por sí sola.
  En cualquier punto de salida se garantiza devolver la mejor solución
  encontrada hasta ese momento.

Parámetros:
    NS         : tamaño de la población (número de soluciones). Default 20.
    F          : factor de escala de mutación ∈ (0, 2]. Default 0.8.
    CR         : tasa de cruce ∈ [0, 1]. Default 0.9.
    time_limit : segundos máximos totales. Default 3600 (1 h).
    seed       : semilla aleatoria. Default 42.

Pseudocódigo seguido (del material del curso):
    for i = 1 to NS:
        P_i = generate_solution(rand)
    end
    for i = 1 to generations:
        for j = 1 to NS:
            a ← rand(1,NS) con a≠j
            b ← rand(1,NS) con a≠b≠j
            c ← rand(1,NS) con a≠b≠c≠j
            r ← rand(1,D)
            for k = 1 to D:
                if rand(k) ≤ CR or k==r:
                    v_j[k] ← P_c[k] + F*(P_a[k] − P_b[k])
                else:
                    v_j[k] ← P_j[k]
            end
            if f(v_j) < f(P_j): P_j ← v_j
        end
    end
"""

import time
import numpy as np

from constructive import ConstructiveAlgorithm
from read_instances import calculate_flow_time
from vnd import evaluate_sequence


# ---------------------------------------------------------------------------
# Codificación / decodificación
# ---------------------------------------------------------------------------

def _keys_to_sequence(keys):
    """Convierte vector de claves reales a secuencia de trabajos (argsort)."""
    return list(np.argsort(keys))


def _evaluate_keys(keys, algo):
    """Evalúa un vector de claves reales: lo decodifica y calcula flow time."""
    seq = _keys_to_sequence(keys)
    flow, starts = evaluate_sequence(seq, algo)
    return flow, starts


# ---------------------------------------------------------------------------
# Clase principal DESearch
# ---------------------------------------------------------------------------

class DESearch:
    """
    Evolución Diferencial con codificación de claves aleatorias para NWJSSP.

    Parámetros
    ----------
    NS         : int   tamaño de población                    (default 20)
    F          : float factor de escala de mutación ∈ (0,2]  (default 0.8)
    CR         : float tasa de cruce ∈ [0,1]                 (default 0.9)
    patience   : int   generaciones sin mejora antes de parar (default 50)
    time_limit : float segundos máximos totales               (default 3600)
    seed       : int   semilla aleatoria                      (default 42)
    """

    def __init__(self, n, m, operations, release_dates,
                 NS: int = 20,
                 F: float = 0.8,
                 CR: float = 0.9,
                 patience: int = 50,
                 time_limit: float = 3600.0,
                 seed: int = 42):
        self.n = n
        self.m = m
        self.operations = operations
        self.release_dates = release_dates
        self.NS = NS
        self.F = F
        self.CR = CR
        self.patience = patience   # generaciones sin mejora antes de parar
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
        Genera NS individuos como vectores de claves reales en [0, 1].
        Si se provee initial_solution, el primer individuo se construye
        a partir de ella.
        Verifica el deadline antes de evaluar cada individuo: si se agota
        el tiempo, detiene la inicialización y devuelve lo generado hasta
        ese momento junto con el mejor encontrado.
        Retorna: (pop, fitness, starts_list, best_flow, best_starts, n_evals)
        """
        rng = np.random.default_rng(self.seed)
        pop = rng.random((self.NS, self.n))

        if initial_solution is not None:
            order = sorted(range(self.n), key=lambda j: initial_solution[j])
            keys0 = np.zeros(self.n)
            for rank, job in enumerate(order):
                keys0[job] = rank / self.n
            pop[0] = keys0

        fitness = np.full(self.NS, np.inf)
        starts_list = [None] * self.NS
        n_evals = 0

        best_flow = np.inf
        best_starts = None

        for i in range(self.NS):
            # Verificar tiempo ANTES de cada evaluación
            if time.time() >= deadline:
                break
            f, s = _evaluate_keys(pop[i], self._algo)
            fitness[i] = f
            starts_list[i] = s
            n_evals += 1
            if f < best_flow:
                best_flow = f
                best_starts = s

        # Si ningún individuo fue evaluado (tiempo agotado antes de empezar),
        # usar la solución inicial directamente como fallback.
        if best_starts is None and initial_solution is not None:
            best_starts = initial_solution
            from read_instances import calculate_flow_time
            best_flow, _ = calculate_flow_time(
                initial_solution, self.operations, self.release_dates
            )

        return pop, fitness, starts_list, best_flow, best_starts, n_evals

    # ------------------------------------------------------------------
    # Operadores DE
    # ------------------------------------------------------------------

    def _mutate(self, pop, j, rng):
        """Mutación DE/rand/1: v = P_c + F*(P_a - P_b)"""
        candidates = [i for i in range(self.NS) if i != j]
        a, b, c = rng.choice(candidates, size=3, replace=False)
        return pop[c] + self.F * (pop[a] - pop[b])

    def _crossover(self, target, mutant, rng):
        """
        Cruce binomial uniforme:
            u[k] = v[k] si rand() ≤ CR o k == r
                   P_j[k] en caso contrario
        """
        D = self.n
        r = int(rng.integers(0, D))
        mask = rng.random(D) <= self.CR
        mask[r] = True
        return np.where(mask, mutant, target)

    @staticmethod
    def _clip(v):
        """Mantiene las claves dentro de [0, 1]."""
        return np.clip(v, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def solve(self, initial_solution=None):
        """
        Ejecuta DE siguiendo el pseudocódigo del curso.
        Respeta el time_limit en TODAS las fases:
          - Inicialización: sale temprano si se agota el tiempo,
            garantizando siempre una solución válida.
          - Generaciones: verifica deadline antes de cada individuo j.
          - Salida: siempre devuelve la mejor solución encontrada
            hasta el momento del corte.

        Returns
        -------
        job_start_times  : list[int]  tiempos de inicio de cada trabajo
        flow_time        : float      suma de tiempos de completación
        computation_time : float      tiempo de cómputo en milisegundos
        n_evaluations    : int        total de evaluaciones de la función objetivo
        """
        start_t = time.time()
        deadline = start_t + self.time_limit
        rng = np.random.default_rng(self.seed)

        # ── Fase 1: Población inicial (con control de tiempo) ──────────
        pop, fitness, starts_list, best_flow, best_starts, n_evaluations = \
            self._init_population(initial_solution, deadline)

        # Si el tiempo se agotó durante la inicialización, retornar ya.
        if time.time() >= deadline:
            computation_time = (time.time() - start_t) * 1000
            return best_starts, best_flow, computation_time, n_evaluations

        best_idx = int(np.argmin(fitness))

        # ── Fase 2: Bucle de generaciones ─────────────────────────────
        generation = 0
        no_improve_count = 0   # generaciones consecutivas sin mejora global

        while time.time() < deadline:
            generation += 1
            improved_this_gen = False

            for j in range(self.NS):
                # Verificar tiempo ANTES de procesar cada individuo
                if time.time() >= deadline:
                    break

                # Mutación + clipping
                mutant = self._clip(self._mutate(pop, j, rng))

                # Cruce
                trial = self._crossover(pop[j], mutant, rng)

                # Evaluación del trial
                trial_flow, trial_starts = _evaluate_keys(trial, self._algo)
                n_evaluations += 1

                # Selección greedy: reemplaza si mejora
                if trial_flow < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_flow
                    starts_list[j] = trial_starts

                    # Actualizar mejor global
                    if trial_flow < best_flow:
                        best_flow = trial_flow
                        best_starts = trial_starts
                        best_idx = j
                        improved_this_gen = True

            # Criterio de parada por no mejora
            if improved_this_gen:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    break

        computation_time = (time.time() - start_t) * 1000
        return best_starts, best_flow, computation_time, n_evaluations