"""Compiled (``@njit``) chemistry kernel: the element-potential equilibrium engine.

This is the single equilibrium and thermodynamics engine of the thermochemistry.  It
consumes the canonical **9-term** NASA arrays that ``SpeciesLibrary.nasa9_arrays()`` packs
(NASA-7 Cantera YAML and NASA-9 NASA Glenn / CEA ``thermo.inp`` data reduced to the same
representation), solves chemical equilibrium (HP and TP), and evaluates species/mixture
thermodynamics and the equilibrium speed of sound.  Being numba-compiled, it runs both
inside the network's ``@njit`` residual/Jacobian loop (via :mod:`nefes.thermo.edge_state`)
and behind the public ``Thermo`` / ``equilibrate_HP`` / ``mixture_properties`` API (via
:mod:`nefes.thermo.equilibrate` and :mod:`nefes.thermo.species`), so the thermochemistry has
one implementation throughout.  Correctness is pinned against Cantera
(``test_cantera_validation``).

Canonical 9-term NASA form (per temperature interval, coefficients
``[a1..a7, b1, b2]``):

    cp/R = a1 T^-2 + a2 T^-1 + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
    h/RT = -a1 T^-2 + a2 ln(T)/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4
           + a7 T^4/5 + b1/T
    s/R  = -a1 T^-2/2 - a2 T^-1 + a3 ln(T) + a4 T + a5 T^2/2 + a6 T^3/3
           + a7 T^4/4 + b2

A NASA-7 polynomial is the special case ``a1 = a2 = 0``, so the same evaluator
serves both data sources.  Units are ``mol`` throughout: the universal gas constant
is per mole, moles ``n_j`` are [mol/kg], element weights and molar masses [kg/mol].

Differentiation contract (shared with :mod:`nefes.thermo.perfect_gas`): the Newton
equilibrium loop branches (damping, convergence, interval choice), so it runs in
**real** arithmetic; complex-step seeds on ``(b0, h, p)`` are attached afterward by
the implicit-function theorem through the converged reduced matrix.  Interval
selection branches only on ``T.real`` so the chosen polynomial stays complex-analytic,
which also makes the species-thermodynamics and frozen-property path (used by
:func:`mixture_properties` and ``SpeciesLibrary.cp_R`` and friends) fully
complex-transparent.
"""

import numpy as np
from numba import njit, types
from numba.extending import overload

from nefes.thermo.constants import R_UNIVERSAL as RU  # universal gas constant [J/(mol*K)]

# Equilibrium-solver controls.
MAX_ITER = 300
TOL = 1.0e-11
TRACE = 1.0e-8  # mole-fraction threshold: "major" vs "trace" species
LN_TRACE_CAP = 9.2103404  # -ln(1e-4): caps trace-species growth per step
# Floor on a species mole fraction inside ``log``: at low temperature an active species'
# moles can underflow to exactly zero, and ``log(0) = -inf`` would then poison the
# element/energy sums through ``n_j f_j = 0 * (-inf) = NaN``.  Flooring the argument keeps
# ``f_j`` finite; the vanishing moles still contribute ~0 to every sum.
MOLE_FRAC_FLOOR = 1.0e-300
# A condensed phase is admitted to the product set when forming it would lower the Gibbs
# energy, i.e. its potential undercuts the element-potential combination by this margin
# (dimensionless g/RT); dropped again when its moles vanish.
CONDENSED_INCLUDE_TOL = 1.0e-8


# ---------------------------------------------------------------------------
# Canonical 9-term species thermodynamics
# ---------------------------------------------------------------------------
@njit(cache=True)
def species_thermo9(coeffs, Tint, T, cpR, hRT, gRT):
    """Fill ``cp/R, h/RT, g/RT`` for every species at temperature ``T``.

    Parameters
    ----------
    coeffs : numpy.ndarray
        Canonical 9-term coefficients, shape ``(Ns, MI, 9)``.
    Tint : numpy.ndarray
        Interior temperature breakpoints, shape ``(Ns, MI-1)``, padded with ``+inf``.
    T : float or complex
        Temperature [K].  The interval is chosen on ``T.real`` so a complex
        perturbation never changes it.
    cpR, hRT, gRT : numpy.ndarray
        Output buffers of length ``Ns`` for ``cp/R``, ``h/RT`` and ``g/RT``; their
        dtype follows ``T``.
    """
    Ns = coeffs.shape[0]
    m = Tint.shape[1]
    Tr = T.real
    lnT = np.log(T)
    Tinv = 1.0 / T
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    for j in range(Ns):
        k = 0
        for q in range(m):
            if Tint[j, q] <= Tr:
                k += 1
        a1 = coeffs[j, k, 0]
        a2 = coeffs[j, k, 1]
        a3 = coeffs[j, k, 2]
        a4 = coeffs[j, k, 3]
        a5 = coeffs[j, k, 4]
        a6 = coeffs[j, k, 5]
        a7 = coeffs[j, k, 6]
        b1 = coeffs[j, k, 7]
        b2 = coeffs[j, k, 8]
        cp = a1 * Tinv2 + a2 * Tinv + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4
        h = (
            -a1 * Tinv2
            + a2 * lnT * Tinv
            + a3
            + a4 * T / 2.0
            + a5 * T2 / 3.0
            + a6 * T3 / 4.0
            + a7 * T4 / 5.0
            + b1 * Tinv
        )
        s = -a1 * Tinv2 / 2.0 - a2 * Tinv + a3 * lnT + a4 * T + a5 * T2 / 2.0 + a6 * T3 / 3.0 + a7 * T4 / 4.0 + b2
        cpR[j] = cp
        hRT[j] = h
        gRT[j] = h - s


@njit(cache=True)
def _mix_cp_h(coeffs, Tint, nj, T):
    """Mixture ``(Σ n_j cp_j/R, Σ n_j h_j/RT)`` at ``T`` (scalar, dtype-generic)."""
    Ns = coeffs.shape[0]
    m = Tint.shape[1]
    Tr = T.real
    lnT = np.log(T)  # unused for cp/h but keeps one code path; cheap
    Tinv = 1.0 / T
    Tinv2 = Tinv * Tinv
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    sum_ncp = nj[0] * 0.0
    sum_nh = nj[0] * 0.0
    for j in range(Ns):
        k = 0
        for q in range(m):
            if Tint[j, q] <= Tr:
                k += 1
        a1 = coeffs[j, k, 0]
        a2 = coeffs[j, k, 1]
        a3 = coeffs[j, k, 2]
        a4 = coeffs[j, k, 3]
        a5 = coeffs[j, k, 4]
        a6 = coeffs[j, k, 5]
        a7 = coeffs[j, k, 6]
        b1 = coeffs[j, k, 7]
        cp = a1 * Tinv2 + a2 * Tinv + a3 + a4 * T + a5 * T2 + a6 * T3 + a7 * T4
        h = (
            -a1 * Tinv2
            + a2 * lnT * Tinv
            + a3
            + a4 * T / 2.0
            + a5 * T2 / 3.0
            + a6 * T3 / 4.0
            + a7 * T4 / 5.0
            + b1 * Tinv
        )
        sum_ncp += nj[j] * cp
        sum_nh += nj[j] * h
    return sum_ncp, sum_nh


# ---------------------------------------------------------------------------
# Element-potential HP equilibrium (CEA, Gordon & McBride NASA RP-1311)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _cea_lambda(nj, ntot, dln_nj, dln_n, dln_T):
    """CEA correction-damping factor in ``(0, 1]`` (RP-1311 eqs 3.1-3.3)."""
    Ns = nj.shape[0]
    amax = 5.0 * abs(dln_n)
    a2 = 5.0 * abs(dln_T)
    if a2 > amax:
        amax = a2
    for j in range(Ns):
        if nj[j] / ntot >= TRACE:
            a = abs(dln_nj[j])
            if a > amax:
                amax = a
    lam = 1.0
    if amax > 2.0:
        lam = 2.0 / amax
    for j in range(Ns):
        x = nj[j] / ntot
        if x < TRACE and dln_nj[j] > 0.0:
            denom = dln_nj[j] - dln_n
            if denom > 1.0e-300:
                lt = (-np.log(x) - LN_TRACE_CAP) / denom
                if 0.0 < lt < lam:
                    lam = lt
    return lam


@njit(cache=True)
def equilibrate_hp(coeffs, Tint, Af, b0, h_target, p, p_ref, T_init, ng, nj, Mout):
    """Solve HP equilibrium in place (gas + condensed); return ``(T, ntot, flag, nit)``.

    Species ``0..ng-1`` are gaseous; ``ng..Ns-1`` are pure condensed phases at unit activity
    (``mu_c/RT = g_c/RT``, no mole-fraction or pressure term), whose moles are direct Newton
    unknowns.  ``ntot`` returned is the **gas** mole total (a condensed phase carries no
    volume), so the caller's ``rho = p/(R ntot T)`` is the bulk gas density.  ``nj`` is
    in/out (warm start -> converged moles [mol/kg]); ``Mout`` receives the converged reduced
    ``(Ne+2+Nc)x(Ne+2+Nc)`` matrix for the IFT seed.  A condensed phase is included when it
    lowers the Gibbs energy (``g_c/RT < sum_i a_ic pi_i``) and dropped when its moles vanish;
    the phase set is decided on the real state, so the complex-step derivative stays exact.
    """
    Ne = Af.shape[0]
    Ns = Af.shape[1]
    Nc = Ns - ng
    dim = Ne + 2 + Nc

    cpR = np.empty(Ns)
    hRT = np.empty(Ns)
    gRT = np.empty(Ns)
    fj = np.empty(Ns)
    dln_nj = np.empty(Ns)

    M = np.zeros((dim, dim))
    rhs = np.zeros(dim)
    included = np.zeros(Nc, dtype=np.bool_)

    # Element / species compaction (CEA keep_el-keep_sp, located on the real abundance): an
    # element with zero gram-atoms carries no products, so its balance row is null -> singular.
    # Drop such elements and every species containing one.  All elements present -> a no-op.
    bscale = 0.0
    for i in range(Ne):
        if b0[i] > bscale:
            bscale = b0[i]
    active_el = np.empty(Ne, dtype=np.bool_)
    for i in range(Ne):
        active_el[i] = b0[i] > 1.0e-13 * bscale
    active_sp = np.empty(Ns, dtype=np.bool_)
    n_active_gas = 0
    for j in range(Ns):
        ok = True
        for i in range(Ne):
            if not active_el[i] and Af[i, j] != 0.0:
                ok = False
                break
        active_sp[j] = ok
        if ok and j < ng:
            n_active_gas += 1
    for j in range(Ns):
        if not active_sp[j]:
            nj[j] = 0.0
    # Seed the condensed phase set from any warm-started positive moles.
    for c in range(Nc):
        if nj[ng + c] > 0.0 and active_sp[ng + c]:
            included[c] = True
        else:
            nj[ng + c] = 0.0
            included[c] = False

    ntot = 0.0
    for j in range(ng):
        ntot += nj[j]
    T = T_init

    # Uniform cold guess (well-conditioned start): gram-atoms spread over the active gas species.
    sb0 = 0.0
    for i in range(Ne):
        sb0 += b0[i]
    uniform = sb0 / (2.0 * n_active_gas)
    if ntot <= 0.0:  # no (or fully condensed) warm start -> seed the gas uniformly
        for j in range(ng):
            nj[j] = uniform if active_sp[j] else 0.0
        ntot = n_active_gas * uniform

    flag = 0
    nit = 0
    n_reset = 0
    for it in range(MAX_ITER):
        nit = it + 1
        species_thermo9(coeffs, Tint, T, cpR, hRT, gRT)
        lnp = np.log(p / p_ref)
        for j in range(ng):
            if active_sp[j]:
                # floor the mole fraction inside log so an underflowed active species stays finite
                xj = nj[j] / ntot
                if xj < MOLE_FRAC_FLOOR:
                    xj = MOLE_FRAC_FLOOR
                fj[j] = gRT[j] + np.log(xj) + lnp
            else:
                fj[j] = 0.0
        hhat_target = h_target / (RU * T)

        for a in range(dim):
            rhs[a] = 0.0
            for b in range(dim):
                M[a, b] = 0.0

        # gas sums
        sum_n = 0.0
        sum_nh = 0.0
        sum_nf = 0.0
        sum_nhf = 0.0
        ccoef = 0.0
        for j in range(ng):
            sum_n += nj[j]
            sum_nh += nj[j] * hRT[j]
            sum_nf += nj[j] * fj[j]
            sum_nhf += nj[j] * hRT[j] * fj[j]
            ccoef += nj[j] * (cpR[j] + hRT[j] * hRT[j])
        # condensed enthalpy (energy balance) and heat capacity (energy diagonal)
        sum_nh_cond = 0.0
        for c in range(Nc):
            if included[c]:
                jc = ng + c
                sum_nh_cond += nj[jc] * hRT[jc]
                ccoef += nj[jc] * cpR[jc]

        # element-balance rows (gas coefficients; condensed columns; total-abundance rhs)
        for i in range(Ne):
            bi = 0.0
            bih = 0.0
            rj = 0.0
            for k in range(Ne):
                s = 0.0
                for j in range(ng):
                    s += Af[i, j] * Af[k, j] * nj[j]
                M[i, k] = s
            for j in range(ng):
                aij_nj = Af[i, j] * nj[j]
                bi += aij_nj
                bih += aij_nj * hRT[j]
                rj += aij_nj * fj[j]
            M[i, Ne] = bi
            M[i, Ne + 1] = bih
            b_cond = 0.0
            for c in range(Nc):
                if included[c]:
                    a_ic = Af[i, ng + c]
                    M[i, Ne + 2 + c] = a_ic
                    b_cond += a_ic * nj[ng + c]
            rhs[i] = b0[i] - (bi + b_cond) + rj

        # total-mole row (gas only)
        for k in range(Ne):
            bk = 0.0
            bkh = 0.0
            for j in range(ng):
                bk += Af[k, j] * nj[j]
                bkh += Af[k, j] * nj[j] * hRT[j]
            M[Ne, k] = bk
            M[Ne + 1, k] = bkh
        M[Ne, Ne] = sum_n - ntot
        M[Ne, Ne + 1] = sum_nh
        rhs[Ne] = ntot - sum_n + sum_nf

        # energy (HP) row: total enthalpy = gas + condensed
        M[Ne + 1, Ne] = sum_nh
        M[Ne + 1, Ne + 1] = ccoef
        rhs[Ne + 1] = hhat_target - (sum_nh + sum_nh_cond) + sum_nhf
        for c in range(Nc):
            if included[c]:
                M[Ne + 1, Ne + 2 + c] = hRT[ng + c]

        # condensed equilibrium rows: g_c/RT = sum_i a_ic pi_i (- h_c/RT dln_T)
        for c in range(Nc):
            r = Ne + 2 + c
            if included[c]:
                for i in range(Ne):
                    M[r, i] = Af[i, ng + c]
                M[r, Ne + 1] = hRT[ng + c]
                rhs[r] = gRT[ng + c]
            else:
                M[r, r] = 1.0
                rhs[r] = 0.0

        # decouple every dropped element (its balance row is null -> identity)
        for i in range(Ne):
            if not active_el[i]:
                for k in range(dim):
                    M[i, k] = 0.0
                    M[k, i] = 0.0
                M[i, i] = 1.0
                rhs[i] = 0.0

        singular = False
        try:
            x = np.linalg.solve(M, rhs)
        except Exception:
            singular = True
        if singular:
            # A warm seed made the reduced matrix singular.  Re-seed to the uniform cold guess
            # (gas) with every condensed phase dropped, and retry; capped so a genuinely
            # rank-deficient system still terminates.
            if n_reset >= 2:
                break
            n_reset += 1
            for j in range(ng):
                nj[j] = uniform if active_sp[j] else 0.0
            for c in range(Nc):
                nj[ng + c] = 0.0
                included[c] = False
            ntot = n_active_gas * uniform
            T = T_init
            continue

        dln_n = x[Ne]
        dln_T = x[Ne + 1]

        # Admit any excluded condensed phase whose formation would lower the Gibbs energy
        # (``g_c/RT < sum_i a_ic pi_i``), BEFORE taking the gas step.  With a carbon-rich mixture
        # the gas alone cannot hold the excess element and the gas-only step is infeasible, so the
        # phase must enter first; seed it to absorb about half of its most-constraining element,
        # then rebuild the reduced system with the new phase set.
        admitted = False
        for c in range(Nc):
            if (not included[c]) and active_sp[ng + c]:
                drive = gRT[ng + c]
                for i in range(Ne):
                    drive -= Af[i, ng + c] * x[i]
                if drive < -CONDENSED_INCLUDE_TOL:
                    seed = 0.0
                    for i in range(Ne):
                        if Af[i, ng + c] > 0.0:
                            s = 0.5 * b0[i] / Af[i, ng + c]
                            if seed == 0.0 or s < seed:
                                seed = s
                    included[c] = True
                    nj[ng + c] = seed if seed > 0.0 else 1.0e-6
                    admitted = True
        if admitted:
            continue

        # gas corrections (condensed dln_nj stay 0 -> inert in the CEA damping factor)
        conv = 0.0
        for j in range(Ns):
            dln_nj[j] = 0.0
        for j in range(ng):
            acc = dln_n + hRT[j] * dln_T - fj[j]
            for i in range(Ne):
                acc += Af[i, j] * x[i]
            dln_nj[j] = acc
            w = (nj[j] / ntot) * abs(acc)
            if w > conv:
                conv = w
        # condensed mole-change contributes to the convergence metric
        for c in range(Nc):
            if included[c] and nj[ng + c] > 0.0:
                w = abs(x[Ne + 2 + c]) / nj[ng + c]
                if w > conv:
                    conv = w

        if conv < TOL and abs(dln_T) < TOL and abs(dln_n) < TOL:
            for a in range(dim):
                for b in range(dim):
                    Mout[a, b] = M[a, b]
            flag = 1
            break

        lam = _cea_lambda(nj, ntot, dln_nj, dln_n, dln_T)
        for j in range(ng):
            nj[j] = nj[j] * np.exp(lam * dln_nj[j])
        ntot = ntot * np.exp(lam * dln_n)
        T = T * np.exp(lam * dln_T)
        # condensed moles update directly; a phase that goes non-positive is dropped
        for c in range(Nc):
            if included[c]:
                nj[ng + c] += lam * x[Ne + 2 + c]
                if nj[ng + c] <= 0.0:
                    nj[ng + c] = 0.0
                    included[c] = False

    if flag == 0:
        for a in range(dim):
            for b in range(dim):
                Mout[a, b] = M[a, b]

    ntot = 0.0
    for j in range(ng):
        ntot += nj[j]
    return T, ntot, flag, nit


# ---------------------------------------------------------------------------
# Complex-step seed via the implicit function theorem
# ---------------------------------------------------------------------------
@njit(cache=True)
def _equil_sens(T_r, nj_r, ntot_r, M, Af, coeffs, Tint, b0_imag, h_imag, p_r, p_imag, ng):
    """Imaginary parts ``(dT, dnj[Ns], dntot)`` of the converged equilibrium state.

    ``M`` is the augmented ``(Ne+2+Nc)`` reduced matrix; the pressure terms involve gas
    species only (a condensed phase carries no ``ln p`` in its potential), and the condensed
    equilibrium rows have no direct ``(h, p, b0)`` dependence so their RHS is zero.  Gas
    sensitivities follow the log relation; condensed mole sensitivities are read off directly.
    """
    Ne = Af.shape[0]
    Ns = Af.shape[1]
    Nc = Ns - ng
    dim = Ne + 2 + Nc

    cpR = np.empty(Ns)
    hRT = np.empty(Ns)
    gRT = np.empty(Ns)
    species_thermo9(coeffs, Tint, T_r, cpR, hRT, gRT)

    c = np.zeros(dim)
    sum_nh_gas = 0.0
    for j in range(ng):
        sum_nh_gas += nj_r[j] * hRT[j]
    for i in range(Ne):
        bi = 0.0
        for j in range(ng):
            bi += Af[i, j] * nj_r[j]
        c[i] = b0_imag[i] + (bi / p_r) * p_imag
    c[Ne] = (ntot_r / p_r) * p_imag
    c[Ne + 1] = h_imag / (RU * T_r) + (sum_nh_gas / p_r) * p_imag

    dy = np.linalg.solve(M, c)

    dlnp = p_imag / p_r
    dT = T_r * dy[Ne + 1]
    dntot = ntot_r * dy[Ne]
    dnj = np.empty(Ns)
    for j in range(ng):
        dln = dy[Ne] + hRT[j] * dy[Ne + 1] - dlnp
        for i in range(Ne):
            dln += Af[i, j] * dy[i]
        dnj[j] = nj_r[j] * dln
    for cc in range(Nc):
        dnj[ng + cc] = dy[Ne + 2 + cc]
    return dT, dnj, dntot


def _attach_equil_imag(T_r, nj_r, ntot_r, M, Af, coeffs, Tint, b0, h, p, ng):
    """Return ``(T, nj, ntot)`` with IFT-spliced imaginary parts (dtype-dispatched).

    Pure-Python base path (used if ever called outside numba); the ``@overload``
    below provides the compiled specializations.
    """
    complex_in = np.iscomplexobj(b0) or isinstance(h, complex) or isinstance(p, complex)
    if not complex_in:
        return T_r, nj_r, ntot_r
    b0_imag = np.imag(np.asarray(b0, dtype=np.complex128))
    dT, dnj, dntot = _equil_sens(
        T_r,
        nj_r,
        ntot_r,
        M,
        Af,
        coeffs,
        Tint,
        b0_imag,
        float(np.imag(h)),
        float(np.real(p)),
        float(np.imag(p)),
        ng,
    )
    T = complex(T_r, dT)
    ntot = complex(ntot_r, dntot)
    nj = nj_r.astype(np.complex128)
    for j in range(nj_r.shape[0]):
        nj[j] = complex(nj_r[j], dnj[j])
    return T, nj, ntot


@overload(_attach_equil_imag, inline="always")
def _attach_equil_imag_ovl(T_r, nj_r, ntot_r, M, Af, coeffs, Tint, b0, h, p, ng):
    b0_complex = getattr(b0, "dtype", None) is not None and isinstance(b0.dtype, types.Complex)
    any_complex = b0_complex or isinstance(h, types.Complex) or isinstance(p, types.Complex)

    if any_complex:

        def impl(T_r, nj_r, ntot_r, M, Af, coeffs, Tint, b0, h, p, ng):
            dT, dnj, dntot = _equil_sens(T_r, nj_r, ntot_r, M, Af, coeffs, Tint, b0.imag, h.imag, p.real, p.imag, ng)
            Ns = nj_r.shape[0]
            nj = np.empty(Ns, dtype=np.complex128)
            for j in range(Ns):
                nj[j] = complex(nj_r[j], dnj[j])
            return complex(T_r, dT), nj, complex(ntot_r, dntot)

        return impl

    def impl(T_r, nj_r, ntot_r, M, Af, coeffs, Tint, b0, h, p, ng):
        return T_r, nj_r, ntot_r

    return impl


@njit(cache=True)
def equilibrate_hp_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_init, ng, nj_init):
    """Complex-step-capable HP equilibrium: real solve + IFT-spliced imaginary part.

    For the complex path **all** of ``(b0, h, p, nj_init)`` must be complex128
    (the consumer recovers a whole complex column), even where the imaginary part
    is zero.  ``ng`` is the number of gaseous species (``ng..Ns-1`` are condensed).
    Returns ``(T, nj, ntot, flag, nit)``.
    """
    Ne = Af.shape[0]
    Ns = Af.shape[1]
    dim = Ne + 2 + (Ns - ng)
    nj = nj_init.real.copy()
    b0r = b0.real.copy()
    M = np.zeros((dim, dim))
    T_r, ntot_r, flag, nit = equilibrate_hp(coeffs, Tint, Af, b0r, h.real, p.real, p_ref, float(T_init), ng, nj, M)
    T, njc, ntot = _attach_equil_imag(T_r, nj, ntot_r, M, Af, coeffs, Tint, b0, h, p, ng)
    return T, njc, ntot, flag, nit


@njit(cache=True)
def equil_state_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_init, ng, nj_init):
    """HP equilibrium reduced to ``(T, rho, c_eq, ntot, flag, nit)``; dtype-generic."""
    T, nj, ntot, flag, nit = equilibrate_hp_cs(coeffs, Tint, Af, b0, h, p, p_ref, T_init, ng, nj_init)
    rho = p / (RU * ntot * T)
    c_eq = equilibrium_sound_speed(coeffs, Tint, Af, nj, ntot, T, p, ng)
    return T, rho, c_eq, ntot, flag, nit


# ---------------------------------------------------------------------------
# Fixed-temperature (TP) equilibrium
# ---------------------------------------------------------------------------
@njit(cache=True)
def equilibrate_tp(coeffs, Tint, Af, b0, T, p, p_ref, ng, nj):
    """Solve TP equilibrium (gas + condensed) at fixed ``T`` in place; return ``(ntot, flag, nit)``.

    The fixed-temperature companion of :func:`equilibrate_hp`: the reduced CEA system drops the
    energy row and the ``d ln T`` unknown, leaving the element-potential + total-mole block plus
    one direct mole unknown per included condensed phase (``dim = Ne+1+Nc``).  Species ``0..ng-1``
    are gaseous; ``ng..Ns-1`` are pure condensed phases at unit activity.  ``ntot`` is the gas
    mole total.  Real arithmetic (the public TP path is not complex-stepped).
    """
    Ne = Af.shape[0]
    Ns = Af.shape[1]
    Nc = Ns - ng
    dim = Ne + 1 + Nc

    cpR = np.empty(Ns)
    hRT = np.empty(Ns)
    gRT = np.empty(Ns)
    fj = np.empty(Ns)
    dln_nj = np.empty(Ns)

    M = np.zeros((dim, dim))
    rhs = np.zeros(dim)
    included = np.zeros(Nc, dtype=np.bool_)

    # element / species compaction on the real abundance (mirrors equilibrate_hp)
    bscale = 0.0
    for i in range(Ne):
        if b0[i] > bscale:
            bscale = b0[i]
    active_el = np.empty(Ne, dtype=np.bool_)
    for i in range(Ne):
        active_el[i] = b0[i] > 1.0e-13 * bscale
    active_sp = np.empty(Ns, dtype=np.bool_)
    n_active_gas = 0
    for j in range(Ns):
        ok = True
        for i in range(Ne):
            if not active_el[i] and Af[i, j] != 0.0:
                ok = False
                break
        active_sp[j] = ok
        if ok and j < ng:
            n_active_gas += 1
    for j in range(Ns):
        if not active_sp[j]:
            nj[j] = 0.0
    for c in range(Nc):
        if nj[ng + c] > 0.0 and active_sp[ng + c]:
            included[c] = True
        else:
            nj[ng + c] = 0.0
            included[c] = False

    ntot = 0.0
    for j in range(ng):
        ntot += nj[j]

    sb0 = 0.0
    for i in range(Ne):
        sb0 += b0[i]
    uniform = sb0 / (2.0 * n_active_gas)
    if ntot <= 0.0:
        for j in range(ng):
            nj[j] = uniform if active_sp[j] else 0.0
        ntot = n_active_gas * uniform

    flag = 0
    nit = 0
    n_reset = 0
    for it in range(MAX_ITER):
        nit = it + 1
        species_thermo9(coeffs, Tint, T, cpR, hRT, gRT)
        lnp = np.log(p / p_ref)
        for j in range(ng):
            if active_sp[j]:
                xj = nj[j] / ntot
                if xj < MOLE_FRAC_FLOOR:
                    xj = MOLE_FRAC_FLOOR
                fj[j] = gRT[j] + np.log(xj) + lnp
            else:
                fj[j] = 0.0

        for a in range(dim):
            rhs[a] = 0.0
            for b in range(dim):
                M[a, b] = 0.0

        sum_n = 0.0
        sum_nf = 0.0
        for j in range(ng):
            sum_n += nj[j]
            sum_nf += nj[j] * fj[j]

        # element-balance rows (gas coefficients; condensed columns; total-abundance rhs)
        for i in range(Ne):
            bi = 0.0
            rj = 0.0
            for k in range(Ne):
                s = 0.0
                for j in range(ng):
                    s += Af[i, j] * Af[k, j] * nj[j]
                M[i, k] = s
            for j in range(ng):
                aij_nj = Af[i, j] * nj[j]
                bi += aij_nj
                rj += aij_nj * fj[j]
            M[i, Ne] = bi
            b_cond = 0.0
            for c in range(Nc):
                if included[c]:
                    a_ic = Af[i, ng + c]
                    M[i, Ne + 1 + c] = a_ic
                    b_cond += a_ic * nj[ng + c]
            rhs[i] = b0[i] - (bi + b_cond) + rj

        # total-mole row (gas only)
        for k in range(Ne):
            bk = 0.0
            for j in range(ng):
                bk += Af[k, j] * nj[j]
            M[Ne, k] = bk
        M[Ne, Ne] = sum_n - ntot
        rhs[Ne] = ntot - sum_n + sum_nf

        # condensed equilibrium rows: g_c/RT = sum_i a_ic pi_i
        for c in range(Nc):
            r = Ne + 1 + c
            if included[c]:
                for i in range(Ne):
                    M[r, i] = Af[i, ng + c]
                rhs[r] = gRT[ng + c]
            else:
                M[r, r] = 1.0
                rhs[r] = 0.0

        for i in range(Ne):
            if not active_el[i]:
                for k in range(dim):
                    M[i, k] = 0.0
                    M[k, i] = 0.0
                M[i, i] = 1.0
                rhs[i] = 0.0

        singular = False
        try:
            x = np.linalg.solve(M, rhs)
        except Exception:
            singular = True
        if singular:
            if n_reset >= 2:
                break
            n_reset += 1
            for j in range(ng):
                nj[j] = uniform if active_sp[j] else 0.0
            for c in range(Nc):
                nj[ng + c] = 0.0
                included[c] = False
            ntot = n_active_gas * uniform
            continue
        dln_n = x[Ne]

        # Admit an excluded condensed phase that wants to form before stepping (see equilibrate_hp).
        admitted = False
        for c in range(Nc):
            if (not included[c]) and active_sp[ng + c]:
                drive = gRT[ng + c]
                for i in range(Ne):
                    drive -= Af[i, ng + c] * x[i]
                if drive < -CONDENSED_INCLUDE_TOL:
                    seed = 0.0
                    for i in range(Ne):
                        if Af[i, ng + c] > 0.0:
                            s = 0.5 * b0[i] / Af[i, ng + c]
                            if seed == 0.0 or s < seed:
                                seed = s
                    included[c] = True
                    nj[ng + c] = seed if seed > 0.0 else 1.0e-6
                    admitted = True
        if admitted:
            continue

        conv = 0.0
        for j in range(Ns):
            dln_nj[j] = 0.0
        for j in range(ng):
            acc = dln_n - fj[j]
            for i in range(Ne):
                acc += Af[i, j] * x[i]
            dln_nj[j] = acc
            w = (nj[j] / ntot) * abs(acc)
            if w > conv:
                conv = w
        for c in range(Nc):
            if included[c] and nj[ng + c] > 0.0:
                w = abs(x[Ne + 1 + c]) / nj[ng + c]
                if w > conv:
                    conv = w

        if conv < TOL and abs(dln_n) < TOL:
            flag = 1
            break

        lam = _cea_lambda(nj, ntot, dln_nj, dln_n, 0.0)
        for j in range(ng):
            nj[j] = nj[j] * np.exp(lam * dln_nj[j])
        ntot = ntot * np.exp(lam * dln_n)
        for c in range(Nc):
            if included[c]:
                nj[ng + c] += lam * x[Ne + 1 + c]
                if nj[ng + c] <= 0.0:
                    nj[ng + c] = 0.0
                    included[c] = False

    ntot = 0.0
    for j in range(ng):
        ntot += nj[j]
    return ntot, flag, nit


# ---------------------------------------------------------------------------
# Sound speeds
# ---------------------------------------------------------------------------
@njit(cache=True)
def equilibrium_sound_speed(coeffs, Tint, Af, nj, ntot, T, p, ng):
    """Equilibrium speed of sound [m/s] from the converged TP-sensitivity block.

    Species ``0..ng-1`` are gaseous; any condensed species (``ng..Ns-1``) are treated as
    frozen at acoustic frequency (a solid does not re-form on the wave time scale), so the
    re-equilibration sensitivity spans the gas only, while the condensed mass still loads the
    density through the gas ``ntot``.  This is an approximation for a two-phase sound speed.
    """
    Ne = Af.shape[0]
    Ns = Af.shape[1]
    dt = nj.dtype

    cpR = np.empty(Ns, dtype=dt)
    hRT = np.empty(Ns, dtype=dt)
    gRT = np.empty(Ns, dtype=dt)
    species_thermo9(coeffs, Tint, T, cpR, hRT, gRT)

    dim = Ne + 1
    Msens = np.zeros((dim, dim), dtype=dt)
    rhsT = np.zeros(dim, dtype=dt)
    rhsP = np.zeros(dim, dtype=dt)

    for i in range(Ne):
        for k in range(Ne):
            s = nj[0] * 0.0
            for j in range(ng):
                s += Af[i, j] * Af[k, j] * nj[j]
            Msens[i, k] = s
        bi = nj[0] * 0.0
        bih = nj[0] * 0.0
        for j in range(ng):
            bi += Af[i, j] * nj[j]
            bih += Af[i, j] * nj[j] * hRT[j]
        Msens[i, Ne] = bi
        Msens[Ne, i] = bi
        rhsT[i] = -bih
        rhsP[i] = bi
    Msens[Ne, Ne] = nj[0] * 0.0
    sum_nh = nj[0] * 0.0
    for j in range(ng):
        sum_nh += nj[j] * hRT[j]
    rhsT[Ne] = -sum_nh
    rhsP[Ne] = ntot

    # Decouple elements with no appreciable gas products (an inert with zero abundance, or an
    # element locked entirely in a frozen condensed phase): their sensitivity row is null ->
    # identity it, mirroring the HP compaction so the block stays non-singular.  Real moles.
    maxn = 0.0
    for j in range(ng):
        nr = nj[j].real
        if nr > maxn:
            maxn = nr
    floor = 1.0e-30 * maxn
    for i in range(Ne):
        active = False
        for j in range(ng):
            if Af[i, j] != 0.0 and nj[j].real > floor:
                active = True
                break
        if not active:
            for k in range(dim):
                Msens[i, k] = nj[0] * 0.0
                Msens[k, i] = nj[0] * 0.0
            Msens[i, i] = nj[0] * 0.0 + 1.0
            rhsT[i] = nj[0] * 0.0
            rhsP[i] = nj[0] * 0.0

    xT = np.linalg.solve(Msens, rhsT)
    xP = np.linalg.solve(Msens, rhsP)

    dlnn_dlnT = xT[Ne]
    dlnn_dlnP = xP[Ne]
    dVdT = 1.0 + dlnn_dlnT
    dVdP = -1.0 + dlnn_dlnP

    cp_eq = nj[0] * 0.0
    for j in range(ng):
        dln_nj_dlnT = xT[Ne] + hRT[j]
        for i in range(Ne):
            dln_nj_dlnT += Af[i, j] * xT[i]
        cp_eq += nj[j] * cpR[j] + nj[j] * dln_nj_dlnT * hRT[j]
    cp_eq *= RU

    V = RU * ntot * T / p
    Cv = cp_eq + (p * V / T) * dVdT * dVdT / dVdP
    gamma_s = cp_eq / Cv
    a2 = -(p * V) * gamma_s / dVdP
    return np.sqrt(a2)


# ---------------------------------------------------------------------------
# Frozen (non-reacting) real-gas state -- the unburnt side of a flame
# ---------------------------------------------------------------------------
@njit(cache=True)
def _frozen_T_real(coeffs, Tint, nj, h_target, T_init):
    """Solve ``Σ n_j H_j(T) = h_target`` for ``T`` (real Newton, f'(T)=cp_mix)."""
    T = T_init
    for _ in range(100):
        sum_ncp, sum_nh = _mix_cp_h(coeffs, Tint, nj, T)
        h = RU * T * sum_nh
        cp_mix = RU * sum_ncp
        f = h - h_target
        dT = -f / cp_mix
        T += dT
        if abs(dT) <= 1e-12 * T:
            break
    return T


def _attach_frozen_imag(T_r, cp_mix, imseed, h, n_feed):
    """Splice the frozen ``T``'s imaginary part ``imseed / cp_mix`` (dtype-dispatched).

    ``imseed = Im(h) - Σ_k H_k(T_r) Im(n_feed_k)`` is the implicit-function seed for
    the temperature when both the enthalpy ``h`` and the reconstructed feed
    composition ``n_feed`` (a linear function of the transported ``Z``) carry
    complex-step perturbations.  Returns a real ``T`` when nothing is complex.
    """
    if isinstance(h, complex) or np.iscomplexobj(n_feed):
        return complex(T_r, imseed / cp_mix)
    return T_r


@overload(_attach_frozen_imag, inline="always")
def _attach_frozen_imag_ovl(T_r, cp_mix, imseed, h, n_feed):
    nf_complex = getattr(n_feed, "dtype", None) is not None and isinstance(n_feed.dtype, types.Complex)
    if nf_complex or isinstance(h, types.Complex):

        def impl(T_r, cp_mix, imseed, h, n_feed):
            return complex(T_r, imseed / cp_mix)

        return impl

    def impl(T_r, cp_mix, imseed, h, n_feed):
        return T_r

    return impl


@njit(cache=True)
def _frozen_comp_seed(coeffs, Tint, n_feed, h, T_r):
    """``Im(h) - Σ_k H_k(T_r) Im(n_feed_k)`` -- the frozen temperature's IFT seed."""
    Nf = n_feed.shape[0]
    cpR = np.empty(Nf)
    hRT = np.empty(Nf)
    gRT = np.empty(Nf)
    species_thermo9(coeffs, Tint, T_r, cpR, hRT, gRT)
    s = h.imag
    for f in range(Nf):
        Hk = RU * T_r * hRT[f]
        s -= Hk * n_feed[f].imag
    return s


@njit(cache=True)
def frozen_state_from_moles_cs(feed_coeffs, feed_Tint, n_feed, h, p, T_init):
    """Frozen real-gas state ``(T, rho, c_frozen, ntot)`` of the unburnt mixture.

    ``n_feed`` [mol/kg] is the feed-species mole vector of the unburnt mixture --
    the forward blend ``xi @ Nfeed`` of the network's feed streams, formed by the
    caller (:func:`nefes.thermo.edge_state.eq_frozen_state`).  No element inversion
    is involved, so any mixture of co-injected fuels is representable.  Dtype-
    generic in ``(n_feed, h, p)``: complex-step seeds on the composition (carried
    in ``n_feed``) and enthalpy propagate via the temperature's implicit-function
    seed -- the lone non-smooth step is the ``h -> T`` inversion.
    """
    n_real = np.real(n_feed).copy()
    T_r = _frozen_T_real(feed_coeffs, feed_Tint, n_real, h.real, T_init)
    sum_ncp_r, _ = _mix_cp_h(feed_coeffs, feed_Tint, n_real, T_r)
    cp_mix_r = RU * sum_ncp_r
    imseed = _frozen_comp_seed(feed_coeffs, feed_Tint, n_feed, h, T_r)
    T = _attach_frozen_imag(T_r, cp_mix_r, imseed, h, n_feed)

    ntot = n_feed[0] * 0.0
    for f in range(n_feed.shape[0]):
        ntot += n_feed[f]
    R_mix = RU * ntot
    rho = p / (R_mix * T)
    sum_ncp, _ = _mix_cp_h(feed_coeffs, feed_Tint, n_feed, T)
    cp_c = RU * sum_ncp
    gamma = cp_c / (cp_c - R_mix)
    c = np.sqrt(gamma * R_mix * T)
    return T, rho, c, ntot
