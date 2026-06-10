import numpy as np
import math


def Beam3DMatrices(m, EA, EIy, EIz, GJ, Im, NodeCoord, ky,kyz,kzy, kz,
                   rho_w, ao, bo, CD_y, CD_z, a_amp, omega,
                   rho_c, A_tot):
    """
    Inputs:
    m         - structural mass per unit length [kg/m]  (= rho_c * A_tot)
    EA        - axial stiffness [N]
    EIy       - bending stiffness about y-axis [N.m2]
    EIz       - bending stiffness about z-axis [N.m2]
    GJ        - torsional stiffness [N.m2]
    Im        - mass radius of gyration squared [m2]
    NodeCoord - ([xl,yl,zl],[xr,yr,zr])
    ky        - nodal mooring spring stiffness in y [N/m]
    kyz       - nodal mooring spring stiffness in y-direction [N/m]
    kzy       - nodal mooring spring stiffness in z-direction [N/m]
    kz        - nodal mooring spring stiffness in z [N/m]
    rho_w     - water density [kg/m3]
    ao        - ellipse semi-axis in z [m]
    bo        - ellipse semi-axis in y [m]
    CD_y      - drag coefficient y-direction [-]
    CD_z      - drag coefficient z-direction [-]
    a_amp     - wave amplitude for linearized drag [m]
    omega     - wave frequency for linearized drag [rad/s]
    rho_c     - structural density [kg/m3]
    A_tot     - total cross-sectional area [m2]
    """

    # ---------------------------------------------------------------
    # 1 - Element length
    # ---------------------------------------------------------------
    xl, yl, zl = NodeCoord[0]
    xr, yr, zr = NodeCoord[1]
    L  = np.sqrt((xr-xl)**2 + (yr-yl)**2 + (zr-zl)**2)
    L2 = L*L
    L3 = L*L2

    # ---------------------------------------------------------------
    # 2 - Transformation matrix T (XY-plane rotation only)
    # ---------------------------------------------------------------
    C = (xr-xl)/L
    S = (yr-yl)/L
    T3 = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])

    if abs(zr - zl) > 1e-6:
        print("Error: Only supports rotation in XY plane")
        T3 = np.eye(3)

    T6  = np.asarray(np.bmat([[T3, np.zeros((3,3))],
                               [np.zeros((3,3)), T3]]))
    T   = np.asarray(np.bmat([[T6, np.zeros((6,6))],
                               [np.zeros((6,6)), T6]]))

    # ---------------------------------------------------------------
    # 3 - Morison parameters
    # ---------------------------------------------------------------
    # Added mass per unit length [kg/m]
    ma_y = rho_w * np.pi * bo**2          # acts on v (y-direction)
    ma_z = rho_w * np.pi * ao**2          # acts on w (z-direction)

    # Inertia coefficients  Cm,y = 1 + bo/ao,  Cm,z = 1 + ao/bo
    Cm_y = 1.0 + bo/ao
    Cm_z = 1.0 + ao/bo

    # Linearized drag coefficients per unit length [N.s/m2]
    #   c_drag = rho_w * C_D * b_o * (8/(3*pi)) * a * omega
    c_y = rho_w * CD_y * bo * (8.0/(3.0*np.pi)) * a_amp * omega
    c_z = rho_w * CD_z * ao * (8.0/(3.0*np.pi)) * a_amp * omega

    # Fluid excitation amplitude per unit length [N.s/m]  (multiplies Vdot)
    #   f_exc = rho_w * pi * ao * bo * Cm * Vdot  +  c_drag * Vdot
    #   => total load operator coefficient:
    # Inertia excitation coefficient per unit length [kg/m]
    # This multiplies water-particle acceleration Vdot.
    p_y = rho_w * np.pi * ao * bo * Cm_y
    p_z = rho_w * np.pi * ao * bo * Cm_z

    # ---------------------------------------------------------------
    # 4 - Hermite shape-function integral template  (L/420 * [...])
    #     Used for consistent mass, damping, and load matrices.
    #     DOF ordering for bending-y block: [v, theta_z, v, theta_z]
    #                                        1   5        7   11
    #     DOF ordering for bending-z block: [w, theta_y, w, theta_y]
    #                                        2   4        8   10
    # ---------------------------------------------------------------
    H_v = np.array([          # 4x4 Hermite sub-matrix (v / theta_z)
        [ 156,    22*L,   54,    -13*L],
        [ 22*L,  4*L2,   13*L,  -3*L2],
        [  54,   13*L,  156,    -22*L],
        [-13*L, -3*L2, -22*L,   4*L2 ]
    ])

    # H_w = np.array([          # 4x4 Hermite sub-matrix (w / theta_y)
    #     [ 156,    22*L,   54,     13*L],
    #     [ 22*L,  4*L2,  -13*L,  -3*L2],
    #     [  54,  -13*L,  156,    -22*L ],   # sign follows theta_y convention?
    #     [ 13*L, -3*L2, -22*L,   4*L2 ]
    # ])

    H_w = np.array([
    [156,     -22*L,   54,      13*L],
    [-22*L,   4*L2,  -13*L,   -3*L2],
    [54,     -13*L,  156,      22*L],
    [13*L,   -3*L2,  22*L,    4*L2]
    ])

    dofs_v = [1, 5, 7, 11]   # v, theta_z at node 1 and 2
    dofs_w = [2, 4, 8, 10]   # w, theta_y at node 1 and 2

    def scatter(mat_12, sub_4x4, dofs):
        """Scatter a 4x4 sub-matrix into a 12x12 matrix."""
        for i in range(4):
            for j in range(4):
                mat_12[dofs[i], dofs[j]] += sub_4x4[i, j]

    # ---------------------------------------------------------------
    # 5 - Stiffness matrix K  (beam + nodal mooring springs)
    # ---------------------------------------------------------------
    K = np.array([
        [EA/L,  0,            0,            0,      0,            0,           -EA/L, 0,            0,            0,      0,            0          ],
        [0,     12*EIz/L3,    0,            0,      0,            6*EIz/L2,    0,    -12*EIz/L3,    0,            0,      0,            6*EIz/L2   ],
        [0,     0,            12*EIy/L3,    0,     -6*EIy/L2,    0,            0,     0,           -12*EIy/L3,    0,     -6*EIy/L2,    0          ],
        [0,     0,            0,            GJ/L,   0,            0,            0,     0,            0,           -GJ/L,   0,            0          ],
        [0,     0,           -6*EIy/L2,     0,      4*EIy/L,     0,            0,     0,            6*EIy/L2,    0,      2*EIy/L,     0          ],
        [0,     6*EIz/L2,    0,            0,      0,            4*EIz/L,      0,    -6*EIz/L2,    0,            0,      0,            2*EIz/L    ],
        [-EA/L, 0,            0,            0,      0,            0,            EA/L,  0,            0,            0,      0,            0          ],
        [0,    -12*EIz/L3,    0,            0,      0,           -6*EIz/L2,    0,     12*EIz/L3,    0,            0,      0,           -6*EIz/L2   ],
        [0,     0,           -12*EIy/L3,    0,      6*EIy/L2,    0,            0,     0,            12*EIy/L3,    0,      6*EIy/L2,    0          ],
        [0,     0,            0,           -GJ/L,   0,            0,            0,     0,            0,            GJ/L,   0,            0          ],
        [0,     0,           -6*EIy/L2,     0,      2*EIy/L,     0,            0,     0,            6*EIy/L2,    0,      4*EIy/L,     0          ],
        [0,     6*EIz/L2,    0,            0,      0,            2*EIz/L,      0,    -6*EIz/L2,    0,            0,      0,            4*EIz/L    ]
    ])

    # Nodal mooring springs (point attachments — no cross-node coupling)
    # K[1, 1] += ky;   K[2, 2] += kz   # node 1
    # K[7, 7] += ky;   K[8, 8] += kz   # node 2
    # K[2, 1] += kyz;  K[7, 8] += kyz
    # K[1, 2] += kzy;  K[8, 7] += kzy
    

    # K[2, 2] += ky;   K[5, 5] += kz   # node 1
    # K[8, 8] += ky;   K[11, 11] += kz   # node 2

    # ---------------------------------------------------------------
    # 6 - Mass matrix M
    #     m_eff differs per direction due to Morison added mass:
    #       axial / torsion : rho_c * A_tot  (= m, no added mass)
    #       v-direction     : m + ma_y
    #       w-direction     : m + ma_z
    # ---------------------------------------------------------------
    M = np.zeros((12, 12))

    # Axial DOFs (0, 6): m only
    M[0, 0] += m*L/420 * 140;   M[0, 6] += m*L/420 * 70
    M[6, 0] += m*L/420 * 70;    M[6, 6] += m*L/420 * 140

    # Torsional DOFs (3, 9): m * Im only
    M[3, 3] += m*L/420 * 140*Im;   M[3, 9]  += m*L/420 * 70*Im
    M[9, 3] += m*L/420 * 70*Im;    M[9, 9]  += m*L/420 * 140*Im

    # v-direction (y): effective mass = m + ma_y
    m_eff_y = m + ma_y
    scatter(M, m_eff_y * L/420 * H_v, dofs_v)

    # w-direction (z): effective mass = m + ma_z
    m_eff_z = m + ma_z
    scatter(M, m_eff_z * L/420 * H_w, dofs_w)

    # ---------------------------------------------------------------
    # 7 - Damping matrix C  (linearized Morison drag only)
    #     c_y, c_z [N.s/m2] — distributed, so use Hermite integral
    # ---------------------------------------------------------------
    C_mat = np.zeros((12, 12))
    scatter(C_mat, c_y * L/420 * H_v, dofs_v)   # drag on v
    scatter(C_mat, c_z * L/420 * H_w, dofs_w)   # drag on w

    # ---------------------------------------------------------------
    # 8 - Load operator Q  (consistent — maps Vdot_y, Vdot_z to nodes)
    #     Q * {Vdot_y, Vdot_z} gives the nodal excitation vector.
    #     p_y, p_z already include both inertia and drag contributions.
    # ---------------------------------------------------------------
    Q = np.zeros((12, 12))
    scatter(Q, p_y * L/420 * H_v, dofs_v)   # excitation in v
    scatter(Q, p_z * L/420 * H_w, dofs_w)   # excitation in w

    # ---------------------------------------------------------------
    # 9 - Rotate all matrices to global coordinates
    # ---------------------------------------------------------------
    def rotate(A):
        return np.matmul(T.T, np.matmul(A, T))

    K     = rotate(K)
    M     = rotate(M)
    C_mat = rotate(C_mat)
    Q     = rotate(Q)

    return M, C_mat, K, Q


# def AddMooringSpringsGlobal(K, mooring_nodes, LDOF, ky, kyz, kzy, kz):
#     """
#     Add discrete mooring spring stiffnesses once to the global stiffness matrix.

#     DOF order per node:
#     [u, v, w, theta_x, theta_y, theta_z]

#     Spring matrix at each mooring node:
#         [Fy]   [ ky   kyz ] [v]
#         [Fz] = [ kzy  kz  ] [w]
#     """

#     for iNode in mooring_nodes:
#         dof_v = iNode * LDOF + 1
#         dof_w = iNode * LDOF + 2

#         K[dof_v, dof_v] += ky
#         K[dof_v, dof_w] += kyz
#         K[dof_w, dof_v] += kzy
#         K[dof_w, dof_w] += kz

#     return K

def AddMooringSpringsGlobal(K, mooring_nodes, LDOF, df_K_des):

    for iNode, (_, row) in zip(mooring_nodes, df_K_des.iterrows()):

        ky  = row["Kxx"]
        kyz = row["Kxz"]
        kzy = row["Kzx"]
        kz  = row["Kzz"]

        dof_v = iNode * LDOF + 1
        dof_w = iNode * LDOF + 2

        K[dof_v, dof_v] += ky
        K[dof_v, dof_w] += kyz
        K[dof_w, dof_v] += kzy
        K[dof_w, dof_w] += kz

    return K

# def AddMooringSpringsGlobal(K, NodeC, LDOF, ky, kyz, kzy, kz, mooring_spacing=25.0):
#     """
#     Add discrete mooring spring stiffnesses at physical mooring positions,
#     snapped to the nearest mesh node.

#     Mooring positions are defined by mooring_spacing in physical coordinates,
#     independent of element size. Each physical mooring location maps to
#     exactly one node (nearest-node snapping).

#     DOF order per node: [u, v, w, theta_x, theta_y, theta_z]

#     Spring matrix:
#         [Fy]   [ ky   kyz ] [v]
#         [Fz] = [ kzy  kz  ] [w]
#     """
#     # Physical x-coordinates of all mooring attachment points
#     x_start  = NodeC[0][0]
#     x_end    = NodeC[-1][0]
#     x_moorings = np.arange(x_start, x_end + mooring_spacing/2, mooring_spacing)

#     # Node x-coordinates
#     x_nodes = np.array([n[0] for n in NodeC])

#     # Snap each mooring to nearest node (unique nodes only)
#     mooring_nodes = set()
#     for xm in x_moorings:
#         i_nearest = np.argmin(np.abs(x_nodes - xm))
#         mooring_nodes.add(i_nearest)

#     # Apply spring stiffness
#     for iNode in mooring_nodes:
#         dof_v = iNode * LDOF + 1
#         dof_w = iNode * LDOF + 2

#         K[dof_v, dof_v] += ky
#         K[dof_v, dof_w] += kyz
#         K[dof_w, dof_v] += kzy
#         K[dof_w, dof_w] += kz

#     return K, sorted(mooring_nodes)