"""
Spatial convergence study – full physical-space model (no ROM).

Strategy
--------
For the convergence metric we want a cheap but representative solve.
The full transient requires integrating through ~400 s of ramp + many
wave periods, which is expensive to repeat for every mesh.  Instead we
evaluate the JONSWAP forcing at a single snapshot time t_snap (chosen
after the ramp has ended, at a wave-crest event) and solve the
frequency-domain quasi-static problem

    K_FF @ u_snap = F_snap

This gives the *stiffness-dominated* spatial displacement pattern and
is sufficient to measure how the discretisation converges with element
length.  A short transient option (n_trans steps with Newmark-β) is
also provided for cases where dynamics matter.

To run the full transient instead, call FEMsolver_full_transient().
"""

import numpy as np
import scipy.linalg as scp
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.integrate as scpi
from scipy.optimize import brentq
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d
import pandas as pd
import time

from threedellips_morison import Beam3DMatrices, AddMooringSpringsGlobal

# ── Helpers shared by all mesh sizes ─────────────────────────────────────────

def wave_number_scalar(omega_i, depth, g=9.81):
    if omega_i <= 0:
        return 0.0
    def residual(k):
        return g * k * np.tanh(k * depth) - omega_i**2
    upper = max(10.0 * omega_i**2 / g + 1.0 / depth, 1e-6)
    while residual(upper) < 0.0:
        upper *= 2.0
    return brentq(residual, 1e-12, upper)


def wave_numbers(omega_array, depth, g=9.81):
    return np.array([wave_number_scalar(om, depth, g=g) for om in omega_array])


def jonswap_spectrum_omega(omega_array, Hs, Tp, gamma=3.3):
    """Numerically normalised JONSWAP S(ω) [m²/(rad/s)]."""
    omega_array = np.asarray(omega_array, dtype=float)
    omega_p_loc = 2.0 * np.pi / Tp
    sigma = np.where(omega_array <= omega_p_loc, 0.07, 0.09)
    r = np.exp(-0.5 * ((omega_array / omega_p_loc - 1.0) / sigma)**2)
    shape = (omega_array**(-5.0)
             * np.exp(-1.25 * (omega_p_loc / omega_array)**4.0)
             * gamma**r)
    target_m0 = (Hs / 4.0)**2
    return shape * target_m0 / trapz(shape, omega_array)


def irregular_wave_kinematics(omega_array, k_array, amplitudes, phases,
                               t_vec, x_wave, z, depth):
    """
    Finite-depth Airy kinematics for an irregular sea state.
    Returns (eta, u_y, u_z, a_y, a_z) each of shape (len(t_vec),).
    """
    theta = (k_array[None, :] * x_wave
             - omega_array[None, :] * t_vec[:, None]
             + phases[None, :])
    C_h = np.cosh(k_array * (depth + z)) / np.sinh(k_array * depth)
    S_v = np.sinh(k_array * (depth + z)) / np.sinh(k_array * depth)
    eta   =  np.sum(amplitudes * np.cos(theta), axis=1)
    u_y   =  np.sum(amplitudes * omega_array * C_h * np.cos(theta), axis=1)
    u_z   =  np.sum(amplitudes * omega_array * S_v * np.sin(theta), axis=1)
    a_y   =  np.sum(amplitudes * omega_array**2 * C_h * np.sin(theta), axis=1)
    a_z   = -np.sum(amplitudes * omega_array**2 * S_v * np.cos(theta), axis=1)
    return eta, u_y, u_z, a_y, a_z


# ── Matrix assembly (mesh-dependent) ─────────────────────────────────────────

def assemble_matrices(Lele, L_tunnel, E, G, mooring_spacing,
                      Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
                      mass, Im, Atot,
                      ky, kz, kyz, kzy,
                      ao, ai, bo, bi, rho_w,
                      CD_y, CD_z, a_tunnel_amp, omega_ref,
                      z_tun, LDOF=6):
    """
    Build global M, C, K, Q matrices for a given element length.

    Returns
    -------
    M_FF, C_FF, K_FF, Q_FF : free-free submatrices (numpy arrays)
    DofsF                  : indices of free DOFs in the global numbering
    NodeC                  : list of node coordinates
    nDofsF                 : number of free DOFs
    """
    nEle = int(round(L_tunnel / Lele))
    nNode = nEle + 1

    TunCX = np.linspace(0, L_tunnel, nNode)
    TunCY = np.zeros(nNode)
    TunCZ = z_tun * np.ones(nNode)
    NodeC = list(zip(TunCX, TunCY, TunCZ))

    nDof = LDOF * nNode
    M = sps.lil_matrix((nDof, nDof))
    C = sps.lil_matrix((nDof, nDof))
    K = sps.lil_matrix((nDof, nDof))
    Q = sps.lil_matrix((nDof, nDof))

    for i in range(nEle):
        n1dof = LDOF * i + np.arange(LDOF)
        n2dof = LDOF * (i + 1) + np.arange(LDOF)
        Me, Ce, Ke, Qe = Beam3DMatrices(
            m=mass, EA=Beam_EA, EIy=Beam_EIy, EIz=Beam_EIz,
            GJ=Beam_GJ, Im=Im,
            NodeCoord=[list(NodeC[i]), list(NodeC[i + 1])],
            ky=ky, kyz=kyz, kzy=kzy, kz=kz,
            rho_w=rho_w, ao=ao, bo=bo,
            CD_y=CD_y, CD_z=CD_z,
            a_amp=a_tunnel_amp, omega=omega_ref,
            rho_c=2500, A_tot=Atot,
        )
        idx = np.append(n1dof, n2dof)
        M[np.ix_(idx, idx)] += Me
        C[np.ix_(idx, idx)] += Ce
        K[np.ix_(idx, idx)] += Ke
        Q[np.ix_(idx, idx)] += Qe

    # Mooring springs
    df_K_des = pd.read_csv("df_K_des.csv")
    x_nodes = TunCX
    x_moorings = np.arange(x_nodes[0], x_nodes[-1] + mooring_spacing / 2,
                           mooring_spacing)
    mooring_nodes = np.array(
        [np.argmin(np.abs(x_nodes - xm)) for xm in x_moorings])
    K = AddMooringSpringsGlobal(K, mooring_nodes, LDOF, df_K_des)

    K = K.tocsc()
    M = M.tocsc()
    C = C.tocsc()
    Q = Q.tocsc()

    # Clamp both ends
    DofsP = np.concatenate([np.arange(LDOF),
                             np.arange(nDof - LDOF, nDof)])
    DofsF = np.setdiff1d(np.arange(nDof), DofsP)
    nDofsF = len(DofsF)

    ff = np.ix_(DofsF, DofsF)
    return (M[ff], C[ff], K[ff], Q[ff],
            DofsF, list(NodeC), nDofsF)


# ── JONSWAP wave setup (mesh-independent) ────────────────────────────────────

def setup_jonswap(Hmo, Tp, d, z_tun, gamma=3.3,
                  omega_min_frac=0.20, omega_max_frac=4.00,
                  n_omega=256, random_seed=123,
                  U_c_y=0.2, U_c_z=0.0):
    """
    Pre-compute JONSWAP frequency components and random phases.
    Returns a dict that can be reused across meshes.
    """
    omega_p = 2.0 * np.pi / Tp
    omega_components = np.linspace(omega_min_frac * omega_p,
                                   omega_max_frac * omega_p, n_omega)
    k_components = wave_numbers(omega_components, d)
    S_omega = jonswap_spectrum_omega(omega_components, Hmo, Tp, gamma)
    domega = np.gradient(omega_components)
    amplitudes = np.sqrt(2.0 * S_omega * domega)
    rng = np.random.default_rng(random_seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_omega)

    Hm0_check = 4.0 * np.sqrt(trapz(S_omega, omega_components))
    print(f"  JONSWAP Hm0 check: {Hm0_check:.3f} m  (target {Hmo:.3f} m)")

    return dict(omega=omega_components, k=k_components,
                amplitudes=amplitudes, phases=phases,
                omega_p=omega_p, Tp=Tp, d=d, z_tun=z_tun,
                U_c_y=U_c_y, U_c_z=U_c_z)


def forcing_at_times(t_vec, wave_params, Q_inertia, Q_drag, DofsF, LDOF=6):
    """
    Evaluate the physical-space force vector F(t) for each t in t_vec.

    Parameters
    ----------
    t_vec       : 1-D array of evaluation times
    wave_params : dict returned by setup_jonswap()
    Q_inertia   : free-free added-mass/inertia matrix  (nDofsF × nDofsF)
    Q_drag      : free-free linearised drag matrix      (nDofsF × nDofsF)
    DofsF       : indices of free DOFs

    Returns
    -------
    F_time : (nDofsF, len(t_vec)) array
    mask_y : boolean mask for y-translation DOFs within DofsF
    mask_z : boolean mask for z-translation DOFs within DofsF
    """
    nDofsF = len(DofsF)
    mask_y = (DofsF % LDOF == 1)
    mask_z = (DofsF % LDOF == 2)

    x_wave = 0.0
    eta_t, V_y_t, V_z_t, Vdot_y_t, Vdot_z_t = irregular_wave_kinematics(
        wave_params['omega'], wave_params['k'],
        wave_params['amplitudes'], wave_params['phases'],
        t_vec, x_wave=x_wave,
        z=wave_params['z_tun'], depth=wave_params['d'],
    )
    V_y_total = V_y_t + wave_params['U_c_y']
    V_z_total = V_z_t + wave_params['U_c_z']

    F_time = np.zeros((nDofsF, len(t_vec)))
    chunk_size = 500
    for j0 in range(0, len(t_vec), chunk_size):
        j1 = min(j0 + chunk_size, len(t_vec))
        nt = j1 - j0
        V_chunk = np.zeros((nDofsF, nt))
        Vdot_chunk = np.zeros((nDofsF, nt))
        V_chunk[mask_y, :]    = V_y_total[j0:j1]
        V_chunk[mask_z, :]    = V_z_total[j0:j1]
        Vdot_chunk[mask_y, :] = Vdot_y_t[j0:j1]
        Vdot_chunk[mask_z, :] = Vdot_z_t[j0:j1]
        F_time[:, j0:j1] = Q_inertia @ Vdot_chunk + Q_drag @ V_chunk

    return F_time, mask_y, mask_z


# ── Option A: single-snapshot static solve (cheapest convergence metric) ──────

def FEMsolver_snapshot(
        Lele,
        # ── structure ──
        L_tunnel, E, G, mooring_spacing,
        Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
        mass, Im, Atot,
        ky, kz, kyz, kzy,
        ao, ai, bo, bi,
        # ── fluid ──
        rho_w, CD_y, CD_z,
        a_tunnel_amp, omega_ref,
        z_tun,
        # ── wave / spectrum ──
        wave_params,         # dict from setup_jonswap()
        # ── snapshot time ──
        t_snap=500.0,        # [s] – must be > t_ramp so ramp=1
):
    """
    Spatial-convergence solve: static equilibrium at a single snapshot.

    Solves  K_FF @ u_snap = F(t_snap)

    This is fast, reproducible, and sufficient to measure how the
    spatial discretisation converges.  Use the same t_snap for every
    mesh so the forcing is identical.

    Returns
    -------
    dict with keys:
        'u_snap'   : displacement vector at free DOFs, shape (nDofsF,)
        'DofsF'    : free-DOF indices
        'NodeC'    : node coordinate list
        'nDofsF'   : number of free DOFs
        'nEle'     : number of elements
        'Lele_act' : actual element length used
        'elapsed'  : wall-clock time [s]
    """
    t0 = time.time()
    nEle = int(round(L_tunnel / Lele))
    Lele_act = L_tunnel / nEle
    print(f"\n[snapshot] Lele={Lele_act:.2f} m  →  {nEle} elements, "
          f"t_snap={t_snap:.1f} s")

    M_FF, C_FF, K_FF, Q_FF, DofsF, NodeC, nDofsF = assemble_matrices(
        Lele, L_tunnel, E, G, mooring_spacing,
        Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
        mass, Im, Atot, ky, kz, kyz, kzy,
        ao, ai, bo, bi, rho_w, CD_y, CD_z,
        a_tunnel_amp, omega_ref, z_tun,
    )

    # Force at one instant (ramp = 1 for t_snap > t_ramp)
    t_vec = np.array([t_snap])
    F_snap, mask_y, mask_z = forcing_at_times(
        t_vec, wave_params, Q_FF, C_FF, DofsF)
    f = F_snap[:, 0]   # shape (nDofsF,)

    # Static solve: sparse for speed on large meshes
    K_sp = sps.csc_matrix(K_FF)
    u_snap = spla.spsolve(K_sp, f)

    elapsed = time.time() - t0
    print(f"  → solved in {elapsed:.1f} s, "
          f"|u_y|_max = {np.max(np.abs(u_snap[mask_y])):.4e} m, "
          f"|u_z|_max = {np.max(np.abs(u_snap[mask_z])):.4e} m")

    return dict(u_snap=u_snap, DofsF=DofsF, NodeC=NodeC,
                nDofsF=nDofsF, nEle=nEle, Lele_act=Lele_act,
                mask_y=mask_y, mask_z=mask_z,
                elapsed=elapsed)


# ── Option B: short transient with Newmark-β (few steps, physical space) ─────

def FEMsolver_short_transient(
        Lele,
        # ── structure ──
        L_tunnel, E, G, mooring_spacing,
        Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
        mass, Im, Atot,
        ky, kz, kyz, kzy,
        ao, ai, bo, bi,
        # ── fluid ──
        rho_w, CD_y, CD_z,
        a_tunnel_amp, omega_ref,
        z_tun,
        # ── wave / spectrum ──
        wave_params,
        # ── time integration ──
        t_start=490.0,       # [s] start of short window (post-ramp)
        n_steps=20,          # number of Newmark steps to take
        dt=0.1,              # time step [s]
        beta=0.25,           # Newmark-β (0.25 = average-acceleration, unconditionally stable)
        gamma_nb=0.5,        # Newmark-γ
):
    """
    Short transient in physical space using Newmark-β.

    Starts from rest at t_start (valid only if t_start > t_ramp so the
    ramp equals 1; the short window means transient start-up is minor).
    For convergence purposes the displacement at t_start + n_steps*dt
    is used.

    Returns the same dict shape as FEMsolver_snapshot, plus
        'u_history'  : (nDofsF, n_steps+1) displacement history
        'tspan_used' : time array
    """
    t0w = time.time()
    nEle = int(round(L_tunnel / Lele))
    Lele_act = L_tunnel / nEle
    print(f"\n[transient] Lele={Lele_act:.2f} m  →  {nEle} elements, "
          f"t_start={t_start:.1f} s, {n_steps} steps")

    M_FF, C_FF, K_FF, Q_FF, DofsF, NodeC, nDofsF = assemble_matrices(
        Lele, L_tunnel, E, G, mooring_spacing,
        Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
        mass, Im, Atot, ky, kz, kyz, kzy,
        ao, ai, bo, bi, rho_w, CD_y, CD_z,
        a_tunnel_amp, omega_ref, z_tun,
    )

    tspan = t_start + np.arange(n_steps + 1) * dt
    F_time, mask_y, mask_z = forcing_at_times(
        tspan, wave_params, Q_FF, C_FF, DofsF)

    # Newmark-β effective stiffness matrix (assembled once)
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma_nb / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = gamma_nb / beta
    K_eff = K_FF + a1 * C_FF + a0 * M_FF
    K_eff_sp = sps.csc_matrix(K_eff)
    K_eff_lu = spla.factorized(K_eff_sp)   # LU once, reuse each step

    u  = np.zeros(nDofsF)
    v  = np.zeros(nDofsF)
    ac = np.linalg.solve(M_FF, F_time[:, 0] - C_FF @ v - K_FF @ u)

    u_history = np.zeros((nDofsF, n_steps + 1))
    u_history[:, 0] = u

    for i in range(n_steps):
        F_next = F_time[:, i + 1]
        rhs = (F_next
               + M_FF @ (a0 * u + a2 * v + (1.0 / (2.0 * beta) - 1.0) * ac)
               + C_FF @ (a1 * u + a3 * v + dt * (gamma_nb / (2.0 * beta) - 1.0) * ac))
        u_new = K_eff_lu(rhs)
        ac_new = a0 * (u_new - u) - a2 * v - (1.0 / (2.0 * beta) - 1.0) * ac
        v_new  = v + dt * ((1.0 - gamma_nb) * ac + gamma_nb * ac_new)
        u, v, ac = u_new, v_new, ac_new
        u_history[:, i + 1] = u

    elapsed = time.time() - t0w
    print(f"  → solved in {elapsed:.1f} s, "
          f"|u_y|_max = {np.max(np.abs(u[mask_y])):.4e} m, "
          f"|u_z|_max = {np.max(np.abs(u[mask_z])):.4e} m")

    return dict(u_snap=u, u_history=u_history, tspan_used=tspan,
                DofsF=DofsF, NodeC=NodeC, nDofsF=nDofsF,
                nEle=nEle, Lele_act=Lele_act,
                mask_y=mask_y, mask_z=mask_z,
                elapsed=elapsed)


# ── Option C: full transient (physical space, no modal reduction) ─────────────

def FEMsolver_full_transient(
        Lele,
        # ── structure ──
        L_tunnel, E, G, mooring_spacing,
        Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
        mass, Im, Atot,
        ky, kz, kyz, kzy,
        ao, ai, bo, bi,
        # ── fluid ──
        rho_w, CD_y, CD_z,
        a_tunnel_amp, omega_ref,
        z_tun,
        # ── wave / spectrum ──
        wave_params,
        # ── time ──
        t_0=0.0,
        n_jonswap=50,
        dt=0.1,
        t_ramp=400.0,
        # ── solver ──
        rtol=1e-4, atol=1e-6,
):
    """
    Full transient solve in physical space with scipy solve_ivp (DOP853).

    The state vector is  y = [u; v]  where u are displacements and v velocities,
    each of length nDofsF.  The ODE is

        M @ v̇ = F(t) − C @ v − K @ u

    written in first-order form.

    The physical-space force F(t) is pre-computed at all tspan points via
    an interpolant, so the ODE right-hand-side is cheap to evaluate.

    Returns
    -------
    dict with:
        'x_phys'   : (nDofsF, n_time) displacement history
        'tspan'    : time vector
        ... same metadata as snapshot solver
    """
    t0w = time.time()
    Tp = wave_params['Tp']
    nEle = int(round(L_tunnel / Lele))
    Lele_act = L_tunnel / nEle
    print(f"\n[full transient] Lele={Lele_act:.2f} m  →  {nEle} elements")

    M_FF, C_FF, K_FF, Q_FF, DofsF, NodeC, nDofsF = assemble_matrices(
        Lele, L_tunnel, E, G, mooring_spacing,
        Beam_EA, Beam_EIy, Beam_EIz, Beam_GJ,
        mass, Im, Atot, ky, kz, kyz, kzy,
        ao, ai, bo, bi, rho_w, CD_y, CD_z,
        a_tunnel_amp, omega_ref, z_tun,
    )

    t_f = t_0 + n_jonswap * Tp
    tspan = np.arange(t_0, t_f + dt, dt)
    n_time = len(tspan)

    # Pre-compute full force history and build interpolants
    print("  Pre-computing force time history …")
    F_time, mask_y, mask_z = forcing_at_times(
        tspan, wave_params, Q_FF, C_FF, DofsF)

    # Per-DOF interpolants (linear is sufficient for dt=0.1 s)
    F_interp = interp1d(tspan, F_time, kind='linear',
                        axis=1, fill_value='extrapolate')

    def ramp(t):
        if t <= t_0:       return 0.0
        if t < t_0 + t_ramp: return (t - t_0) / t_ramp
        return 1.0

    # Pre-factorise M for efficiency (M is constant)
    # Use sparse solver for large systems
    M_sp = sps.csc_matrix(M_FF)
    M_lu = spla.factorized(M_sp)

    y0 = np.zeros(2 * nDofsF)

    def odefun(t, y):
        u = y[:nDofsF]
        v = y[nDofsF:]
        f = ramp(t) * F_interp(t)
        rhs = f - C_FF @ v - K_FF @ u
        vdot = M_lu(rhs)
        return np.concatenate([v, vdot])

    print("  Integrating … (this may take a while for fine meshes)")
    sol = scpi.solve_ivp(
        fun=odefun,
        t_span=[t_0, tspan[-1]],
        y0=y0,
        t_eval=tspan,
        method='DOP853',
        rtol=rtol,
        atol=atol,
    )

    x_phys = sol.y[:nDofsF, :]   # displacements, shape (nDofsF, n_time)

    elapsed = time.time() - t0w
    print(f"  → solved in {elapsed:.1f} s")

    return dict(x_phys=x_phys, tspan=tspan,
                DofsF=DofsF, NodeC=NodeC, nDofsF=nDofsF,
                nEle=nEle, Lele_act=Lele_act,
                mask_y=mask_y, mask_z=mask_z,
                elapsed=elapsed)


# ── Convergence loop and postprocessing ──────────────────────────────────────

def run_convergence_study(
        element_lengths,
        solver_func,          # FEMsolver_snapshot, _short_transient, or _full_transient
        common_kwargs,        # dict of all args except Lele
        x_ref_frac=0.5,       # tunnel fraction for comparison point (0.5 = midspan)
):
    """
    Run solver_func for each element length and collect results.

    Parameters
    ----------
    element_lengths : list of target element lengths [m]
    solver_func     : one of the three FEMsolver_* functions above
    common_kwargs   : keyword arguments forwarded to every solver call
    x_ref_frac      : fractional position along tunnel for comparison DOF

    Returns
    -------
    results : list of result dicts (one per element length)
    summary : dict with arrays 'Lele', 'u_y_ref', 'u_z_ref' for convergence plot
    """
    results = []
    L_tunnel = common_kwargs.get('L_tunnel', 27000.0)

    for Lele in element_lengths:
        res = solver_func(Lele=Lele, **common_kwargs)
        results.append(res)

    # Extract displacement at a reference node for each mesh
    u_y_ref = []
    u_z_ref = []
    Lele_act = []

    for res in results:
        NodeC  = res['NodeC']
        DofsF  = res['DofsF']
        mask_y = res['mask_y']
        mask_z = res['mask_z']

        x_nodes = np.array([nc[0] for nc in NodeC])
        x_ref   = x_ref_frac * L_tunnel
        i_ref   = np.argmin(np.abs(x_nodes - x_ref))

        # Map global node i_ref to a free-DOF index
        # DOF layout: node i → global DOFs [6i .. 6i+5]
        # y-translation = DOF 6i+1, z-translation = DOF 6i+2
        LDOF = 6
        glob_dof_y = LDOF * i_ref + 1
        glob_dof_z = LDOF * i_ref + 2

        # Find position of these global DOFs within DofsF
        idx_y = np.searchsorted(DofsF, glob_dof_y)
        idx_z = np.searchsorted(DofsF, glob_dof_z)

        u_snap = res['u_snap']
        u_y_ref.append(u_snap[idx_y])
        u_z_ref.append(u_snap[idx_z])
        Lele_act.append(res['Lele_act'])

    summary = dict(
        Lele   = np.array(Lele_act),
        u_y_ref = np.array(u_y_ref),
        u_z_ref = np.array(u_z_ref),
        x_ref_frac = x_ref_frac,
    )

    return results, summary


def print_convergence_table(summary):
    """Print a formatted convergence table."""
    print("\n")
    print(f"{'Lele [m]':>12} {'nEle':>8} {'u_y_ref [m]':>16} {'u_z_ref [m]':>16}")
    L = summary['Lele']
    for i, le in enumerate(L):
        nEle = int(round(27000.0 / le))
        print(f"{le:12.1f} {nEle:8d} "
              f"{summary['u_y_ref'][i]:16.6e} "
              f"{summary['u_z_ref'][i]:16.6e}")


# ── Example usage ─────────────────────────────────────────────────────────────
# (Run this block in your notebook after importing from this module,
#  or paste it directly into a notebook cell.)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── Material / section constants (copy from your notebook) ───────────────
    E = 40e9;  G = 12e9
    L_tunnel = 27000.0;  z_tun = -27.5
    mooring_spacing = 25.0
    mass = 266.63e3;  Atot = 104.6242421
    Iy = 3256.17;  Iz = 5950.91
    ao = 14;  ai = 13;  bo = 8.5;  bi = 7.5
    Im = ((ao*bo*(ao**2+bo**2) - ai*bi*(ai**2+bi**2))
          / (4*(ao*bo - ai*bi)))
    Beam_EIy = E*Iy;  Beam_EIz = E*Iz
    Beam_EA  = E*Atot
    J = Iy + Iz;  Beam_GJ = G*J
    ky = 2.03e6;  kz = 10.6e6;  kyz = kzy = 0.102e6
    CD_y = 0.5;  CD_z = 1.5
    rho_w = 1025.0;  g = 9.81;  d = 80.0

    Hmo = 2.965;  Tp = 8.06;  gamma = 3.3
    omega_p = 2.0*np.pi/Tp
    k_wave = wave_number_scalar(omega_p, d)
    a_surface_amp = Hmo/2.0
    a_y_peak = a_surface_amp * np.cosh(k_wave*(d+z_tun)) / np.sinh(k_wave*d)
    a_z_peak = a_surface_amp * np.sinh(k_wave*(d+z_tun)) / np.sinh(k_wave*d)
    a_tunnel_amp = np.sqrt(0.5*(a_y_peak**2 + a_z_peak**2))

    # ── Pre-compute wave field once ───────────────────────────────────────────
    wave_params = setup_jonswap(Hmo, Tp, d, z_tun, gamma=gamma)

    # ── Common kwargs for the solver ─────────────────────────────────────────
    common = dict(
        L_tunnel=L_tunnel, E=E, G=G, mooring_spacing=mooring_spacing,
        Beam_EA=Beam_EA, Beam_EIy=Beam_EIy, Beam_EIz=Beam_EIz,
        Beam_GJ=Beam_GJ, mass=mass, Im=Im, Atot=Atot,
        ky=ky, kz=kz, kyz=kyz, kzy=kzy,
        ao=ao, ai=ai, bo=bo, bi=bi,
        rho_w=rho_w, CD_y=CD_y, CD_z=CD_z,
        a_tunnel_amp=a_tunnel_amp, omega_ref=omega_p,
        z_tun=z_tun,
        wave_params=wave_params,
        t_snap=500.0,    # well past ramp; change as needed
    )

    # ── Convergence study ────────────────────────────────────────────────────
    element_lengths = [500.0, 250.0, 125.0, 62.5, 31.25]

    results, summary = run_convergence_study(
        element_lengths=element_lengths,
        solver_func=FEMsolver_snapshot,
        common_kwargs=common,
    )

    print_convergence_table(summary)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in zip(axes,
                               ['u_y_ref', 'u_z_ref'],
                               ['Horizontal (y) displacement [m]',
                                'Vertical (z) displacement [m]']):
        ax.semilogx(summary['Lele'], np.abs(summary[key]), 'o-')
        ax.set_xlabel('Element length [m]')
        ax.set_ylabel(label)
        ax.set_title(f'Convergence at x = {summary["x_ref_frac"]*L_tunnel:.0f} m')
        ax.invert_xaxis()
        ax.grid(True, which='both')
    plt.tight_layout()
    # plt.savefig('convergence_plot.png', dpi=150)
    plt.show()
