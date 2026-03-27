import mosek.fusion as mf
from abc import ABC, abstractmethod
import numpy as np
from MathWrapper import MathWrapper as mw
from typing import Any
from ModelWrapper import ModelWrapper
import scipy.linalg as sla

# def block(rows):
#     """
#     imitate np.block() method, e.g. block([[A,B],[C,D]]) then return block matrix 
#     """
#     raw_rows = []
#     for row in rows:
#         raw_row = [item.Expr if isinstance(item,MathWrapper) else item for item in rows]
#         raw_rows.append(mf.Expr.hstack(raw_row)) #stack horizontally then
#     return MathWrapper(mf.Expr.vstack(raw_rows)) #stack vertically then get block matrix

class MosekSeigyo(ABC):
    """Parent class: Holds the tools, but waits for dimensions."""
    
    def __init__(self, name: str):
        self.name = name
        self.n = None
        self.I = None
        self.psd_domain = None

    def _setup_geometry(self, n: int):
        """
        Helper: Call this from the child once 'n' is known.
        Creates the reusable MOSEK objects.
        """
        self.n = n
        self.I = mf.Matrix.eye(n)
        self.psd_domain = mf.Domain.inPSDCone(n)

    @abstractmethod
    def solve(self)->Any:
        pass

class LyapunovIneqSolver(MosekSeigyo):
    def __init__(self, A: np.ndarray):
        super().__init__("Lyapunov_Check")
        self.A_np = np.array(A)
        self.n = self.A_np.shape[0]
        self._setup_geometry(self.n)

    def solve_without_both_Wrappers(self):
        with mf.Model(self.name) as M:
            P = M.variable("P", self.psd_domain)
            A = mf.Matrix.dense(self.A_np)
            PA = mf.Expr.mul(P, A)
            Q  = mf.Expr.add(PA, mf.Expr.transpose(PA))
            M.constraint(mf.Expr.sub(P, self.I), self.psd_domain)
            M.constraint(mf.Expr.sub(Q, self.I), self.psd_domain)
            M.solve() 
        
    def solve_without_ModelWrapper(self):
        with mf.Model(self.name) as M:
            P = mw(M.variable("P", self.psd_domain))
            A = mw(mf.Matrix.dense(self.A_np))
            I = mw(self.I)
            Q = P@A + A.T@P
            M.constraint((P>>I).Expr, self.psd_domain)
            M.constraint((Q>>I).Expr, self.psd_domain)
            M.solve()
            status = M.getProblemStatus()
            if status == mf.ProblemStatus.PrimalAndDualFeasible:
                return P.val()
            
    def solve(self):
        with ModelWrapper(self.name) as M:
            P = M.variable("P", self.psd_domain)
            A = M.const_matrix(self.A_np)
            I = mw.eye(self.n)
            Q = P@A + A.T@P
            M.constraint(P>>I)
            M.constraint(Q>>I)
            M.solve()
            status = M.getProblemStatus()
            if status == mf.ProblemStatus.PrimalAndDualFeasible:
                return P.val()
            

class DARE_LMI(MosekSeigyo):
    """A solver class for Discrete Algebraic Riccati Equation under Linear Matrix Inequality Formulation"""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray | int,
        C: np.ndarray | None = None,
        Q_dual: np.ndarray | None = None,
        R_dual: np.ndarray | None = None,
    ) -> None:
        super().__init__("LMI_DiscreteRiccati")
        self.A_np = np.array(A)
        self.B_np = np.array(B)
        self.Q_np = np.array(Q)
        self.R_np = np.array(R)
        self.C_np = np.array(C) if C is not None else None
        self.Q_dual_np = np.array(Q_dual) if Q_dual is not None else None
        self.R_dual_np = np.array(R_dual) if R_dual is not None else None
        self.n = self.A_np.shape[0]
        self.nx = self.A_np.shape[0] #row
        self.nu = self.B_np.shape[1] #col
        self._setup_geometry(self.n)

    def solve(self)->tuple[np.ndarray,np.ndarray]:
        with ModelWrapper(self.name) as M:
            I = mw.eye(self.n)
            W = M.variable("W", self.psd_domain) # W=inv(P)
            Y = M.variable("Y", [self.nu, self.nx]) # Y=KW=K@inv(P)
            A = M.const_matrix(self.A_np)
            B = M.const_matrix(self.B_np)
            Q = M.const_matrix(self.Q_np)
            R = M.const_matrix(self.R_np)
            Ix = mw.eye(self.Q_np.shape[0])
            Iu = mw.eye(self.R_np.shape[0])
            Q_sqrt, R_sqrt = mw.sqrt(Q) , mw.sqrt(R)

            lmi = mw.auto_bmat([ [    W     , W@A.T + Y.T@B.T , W@Q_sqrt , Y.T@R_sqrt],
                                 [A@W + B@Y ,       W         ,    None  ,    None   ],
                                 [ Q_sqrt@W ,      None       ,     Ix   ,    None   ],
                                 [ R_sqrt@Y ,      None       ,    None  ,     Iu    ]  ])
            M.constraint(lmi >> 0)
            M.objective("DARE--LMI", mf.ObjectiveSense.Maximize, W.trace())
            M.solve()
            return W.val() , Y.val()

    def solve_dual(
        self,
        C: np.ndarray | None = None,
        Q_dual: np.ndarray | None = None,
        R_dual: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dual LMI solve using (A^T, C^T). If Q_dual/R_dual are None,
        reuse primal weights with compatible shapes.
        """
        if C is None:
            C = self.C_np
        if C is None:
            raise ValueError("C is required for dual solve but was None.")

        C_np = np.array(C)
        if C_np.ndim != 2 or C_np.shape[1] != self.nx:
            raise ValueError(
                f"C must be 2D with shape (ny, {self.nx}), got {C_np.shape}."
            )

        ny = C_np.shape[0]

        if Q_dual is None:
            Q_dual = self.Q_dual_np
        if Q_dual is None:
            Q_dual_np = np.array(self.Q_np, dtype=float)
        else:
            Q_dual_np = np.array(Q_dual, dtype=float)

        if Q_dual_np.shape != (self.nx, self.nx):
            raise ValueError(
                f"Q_dual must be shape ({self.nx}, {self.nx}), got {Q_dual_np.shape}."
            )

        if R_dual is None:
            R_dual = self.R_dual_np
        if R_dual is None:
            R_dual_np = np.eye(ny)
        else:
            R_dual_np = np.array(R_dual, dtype=float)

        if R_dual_np.shape != (ny, ny):
            raise ValueError(
                f"R_dual must be shape ({ny}, {ny}), got {R_dual_np.shape}."
            )

        with ModelWrapper(f"{self.name}_dual") as M:
            W = M.variable("W", self.psd_domain) # W=inv(P)
            Y = M.variable("Y", [ny, self.nx]) # Y=K_dual*W
            A_t = M.const_matrix(self.A_np.T)
            C_t = M.const_matrix(C_np.T)
            Q = M.const_matrix(Q_dual_np)
            R = M.const_matrix(R_dual_np)
            Ix = mw.eye(Q_dual_np.shape[0])
            Iy = mw.eye(R_dual_np.shape[0])
            Q_sqrt, R_sqrt = mw.sqrt(Q) , mw.sqrt(R)

            lmi = mw.auto_bmat([ [    W     , W@A_t.T + Y.T@C_t.T , W@Q_sqrt , Y.T@R_sqrt],
                                 [A_t@W + C_t@Y ,       W         ,    None  ,    None   ],
                                 [ Q_sqrt@W ,      None       ,     Ix   ,    None   ],
                                 [ R_sqrt@Y ,      None       ,    None  ,     Iy    ]  ])
            M.constraint(lmi >> 0)
            M.objective("DARE--LMI--DUAL", mf.ObjectiveSense.Maximize, W.trace())
            M.solve()
            return W.val(), Y.val()


class CartSystem:
    """
    Physics-based AITL system builder (Python port of build_aitl_system.m).
    Produces discrete-time A, B plus resampled A_tilde, B_tilde.
    """

    def __init__(
        self,
        M: int = 5,
        mi: list[float] | np.ndarray | None = None,
        mp: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        kappa: float = 0.5,
        h: float = 0.2,
        dt: float = 0.03,
        walk: list[int] | np.ndarray | None = None,
        C: np.ndarray | None = None,
    ) -> None:
        self.M = int(M)
        if mi is None:
            mi = [7.0, 6.0, 6.0, 8.0, 5.0, 7.0, 6.0, 6.0, 8.0, 5.0]
        mi = np.array(mi, dtype=float).flatten()
        if mi.size < self.M:
            raise ValueError("mi must have at least M entries.")
        if mi.size > self.M:
            mi = mi[: self.M]

        if walk is None:
            walk = list(range(1, self.M + 1))
        walk = np.array(walk, dtype=int).flatten()

        self.mi = mi
        self.mp = float(mp)
        self.l = float(l)
        self.g = float(g)
        self.kappa = float(kappa)
        self.h = float(h)
        self.dt = float(dt)
        self.walk = walk

        self.A, self.B, self.A_tilde, self.B_tilde = self._build_system()

        self.C_tilde, self.D_tilde_u = self._build_observation_matrices()

        if C is None:
            self.C = np.eye(self.A.shape[0])
        else:
            self.C = np.array(C, dtype=float)

    def _build_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        M = self.M
        mi = self.mi
        mp = self.mp
        l = self.l
        g = self.g
        kappa = self.kappa
        h = self.h
        dt = self.dt
        walk = self.walk

        C = np.zeros(M)
        C1 = np.zeros(M)
        C2 = np.zeros(M)
        C3 = np.zeros(M)
        I = 0.0

        for i in range(M):
            C[i] = 1.0 / ((mi[i] + mp) * (I + mp * l**2) - (mp * l) ** 2)
            C1[i] = (I + mp * l**2) * C[i]
            C2[i] = -mp * l * C[i]
            C3[i] = (mi[i] + mp) * C[i]

        Z = np.zeros((M, M))
        I_n = np.eye(M)

        K = (
            np.diag(-2 * kappa * C1)
            + np.diag(kappa * C1[:-1], 1)
            + np.diag(kappa * C1[1:], -1)
        )
        H = (
            np.diag(-2 * h * C1)
            + np.diag(h * C1[:-1], 1)
            + np.diag(h * C1[1:], -1)
        )
        Kth = (
            np.diag(-2 * kappa * C2)
            + np.diag(kappa * C2[:-1], 1)
            + np.diag(kappa * C2[1:], -1)
        )
        Hth = (
            np.diag(-2 * h * C2)
            + np.diag(h * C2[:-1], 1)
            + np.diag(h * C2[1:], -1)
        )
        Gx = np.diag(g * mp * C2)
        Gth = np.diag(g * mp * C3)

        A_temp = np.block(
            [
                [Z, Z, I_n, Z],
                [Z, Z, Z, I_n],
                [K, Gx, H, Z],
                [Kth, Gth, Hth, Z],
            ]
        )

        B_temp = np.block(
            [
                [np.zeros((M, M))],
                [np.zeros((M, M))],
                [np.diag(C1)],
                [np.diag(C2)],
            ]
        )

        nx = 4 * M
        nu = M
        P_reorder = np.zeros((nx, nx))
        for i in range(M):
            P_reorder[i * 4 + 0, M + i] = 1
            P_reorder[i * 4 + 1, 3 * M + i] = 1
            P_reorder[i * 4 + 2, i] = 1
            P_reorder[i * 4 + 3, 2 * M + i] = 1

        A_cont = P_reorder @ A_temp @ P_reorder.T
        B_cont = P_reorder @ B_temp

        M_aug = np.block(
            [
                [A_cont, B_cont],
                [np.zeros((nu, nx + nu))],
            ]
        )
        M_disc = sla.expm(M_aug * dt)
        A = M_disc[:nx, :nx]
        B = M_disc[:nx, nx : nx + nu]

        N_walk = walk.size
        A_tilde = np.linalg.matrix_power(A, N_walk)

        B_hat = np.zeros((nx, nu * N_walk))
        for i in range(N_walk):
            power_idx = N_walk - i - 1
            B_hat[:, i * nu : (i + 1) * nu] = np.linalg.matrix_power(A, power_idx) @ B

        P_w = np.zeros((nu * N_walk, nu))
        for i in range(N_walk):
            cart_idx = int(walk[i]) - 1
            if cart_idx < 0 or cart_idx >= nu:
                raise ValueError("walk entries must be in [1, M].")
            P_w[i * nu + cart_idx, cart_idx] = 1

        B_tilde = B_hat @ P_w
        return A, B, A_tilde, B_tilde

    def _build_observation_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        A = self.A
        B = self.B
        walk = self.walk
        M = self.M
        nx = A.shape[0]
        nu = B.shape[1]

        ny_per_step = 4
        N_walk = walk.size
        ny_total = ny_per_step * N_walk

        C_tilde = np.zeros((ny_total, nx))
        D_tilde_u = np.zeros((ny_total, nu * N_walk))

        for i in range(N_walk):
            step_cart = int(walk[i])
            indices = (step_cart - 1) * 4 + np.arange(1, 5)
            C_step = np.zeros((ny_per_step, nx))
            C_step[:, indices - 1] = np.eye(ny_per_step)

            row_start = i * ny_per_step
            row_end = row_start + ny_per_step
            power_idx = i

            C_tilde[row_start:row_end, :] = C_step @ np.linalg.matrix_power(A, power_idx)

            for j in range(i):
                col_start = j * nu
                col_end = col_start + nu
                D_val = C_step @ np.linalg.matrix_power(A, i - 1 - j) @ B
                D_tilde_u[row_start:row_end, col_start:col_end] = D_val

        return C_tilde, D_tilde_u


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    blocks = [B]
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = Ak @ A
        blocks.append(Ak @ B)
    return np.concatenate(blocks, axis=1)


def choose_feedback_sign(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    eigs_plus = np.linalg.eigvals(A + B @ K)
    eigs_minus = np.linalg.eigvals(A - B @ K)
    max_plus = float(np.max(np.abs(eigs_plus)))
    max_minus = float(np.max(np.abs(eigs_minus)))
    if max_minus < 1.0 and max_minus < max_plus:
        return -K, max_plus, max_minus
    return K, max_plus, max_minus


def choose_observer_sign(
    A_t: np.ndarray,
    C_t: np.ndarray,
    K_dual: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    eigs_plus = np.linalg.eigvals(A_t + C_t @ K_dual)
    eigs_minus = np.linalg.eigvals(A_t - C_t @ K_dual)
    max_plus = float(np.max(np.abs(eigs_plus)))
    max_minus = float(np.max(np.abs(eigs_minus)))
    if max_minus < 1.0 and max_minus < max_plus:
        return -K_dual, max_plus, max_minus
    return K_dual, max_plus, max_minus


# A = np.array([
#     [ 2 ,-1 , 0],
#     [-1 , 2 ,-1],
#     [ 0 ,-1 , 2]])
# solver = LyapunovIneqSolver(A)
# print(solver.solve())


cart_sys = CartSystem()

Q_diag = np.array([10.0, 1.0, 300.0, 1.0])
Q_val = np.kron(np.eye(cart_sys.M), np.diag(Q_diag))
R_val = np.eye(cart_sys.M)

Q_dual_val = Q_val.copy()
R_dual_val = np.eye(cart_sys.C_tilde.shape[0])

solver = DARE_LMI(
    cart_sys.A_tilde,
    cart_sys.B_tilde,
    Q_val,
    R_val,
    C=cart_sys.C_tilde,
    Q_dual=Q_dual_val,
    R_dual=R_dual_val,
)

try:
    ctrb_rank = np.linalg.matrix_rank(controllability_matrix(cart_sys.A_tilde, cart_sys.B_tilde))
    print(f"Controllability rank: {ctrb_rank} / {cart_sys.A_tilde.shape[0]}")

    W_opt, Y_opt = solver.solve()
    
    # Recover P and K
    P = np.linalg.inv(W_opt)
    K_raw = Y_opt @ P
    K, max_plus, max_minus = choose_feedback_sign(cart_sys.A_tilde, cart_sys.B_tilde, K_raw)
    
    print("Success! Solver finished.")
    print(f"Feedback Gain K:\n{K}")
    print(f"Max |eig(A_tilde + B_tilde*K)|: {max_plus:.6f}")
    print(f"Max |eig(A_tilde - B_tilde*K)|: {max_minus:.6f}")

    W_dual, Y_dual = solver.solve_dual()
    P_dual = np.linalg.inv(W_dual)
    K_dual_raw = Y_dual @ P_dual
    K_dual, max_plus_dual, max_minus_dual = choose_observer_sign(
        cart_sys.A_tilde.T,
        cart_sys.C_tilde.T,
        K_dual_raw,
    )
    L = -K_dual.T
    print(f"Observer Gain L:\n{L}")
    print(f"Max |eig(A_tildeT + C_tildeT*K_dual)|: {max_plus_dual:.6f}")
    print(f"Max |eig(A_tildeT - C_tildeT*K_dual)|: {max_minus_dual:.6f}")

    obs_eigs = np.linalg.eigvals(cart_sys.A_tilde - L @ cart_sys.C_tilde)
    print(f"Observer max |eig|: {float(np.max(np.abs(obs_eigs))):.6f}")
    
except Exception as e:
    print(f"Solver crashed: {e}")


