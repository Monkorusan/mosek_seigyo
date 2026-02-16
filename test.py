import mosek.fusion as mf
from abc import ABC, abstractmethod
import numpy as np
from MathWrapper import MathWrapper as mw
from typing import Any
from ModelWrapper import ModelWrapper

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

    def __init__(self, A: np.ndarray, B: np.ndarray, Q:np.ndarray, R:np.ndarray|int)->None:
        super().__init__("LMI_DiscreteRiccati")
        self.A_np = np.array(A)
        self.B_np = np.array(B)
        self.Q_np = np.array(Q)
        self.R_np = np.array(R)
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


# A = np.array([
#     [ 2 ,-1 , 0],
#     [-1 , 2 ,-1],
#     [ 0 ,-1 , 2]])
# solver = LyapunovIneqSolver(A)
# print(solver.solve())


# --- Physical Constants ---
dt = 0.1  # Time step (100ms)

# --- 1. System Matrices (A, B) ---
# State x = [position; velocity]
# Dynamics: pos += vel*dt; vel += force*dt
A_val = np.array([
    [1.0, dt], 
    [0.0, 1.0]
])

B_val = np.array([
    [0.5 * dt**2],  # affect position (acceleration)
    [dt]            # affect velocity
])

# --- 2. Cost Matrices (Q, R) ---
# High penalty on position error, low on velocity
Q_val = np.diag([10.0, 1.0]) 

# Small penalty on control effort (cheap fuel)
R_val = np.array([[0.1]])

# --- 3. Run Solver ---
solver = DARE_LMI(A_val, B_val, Q_val, R_val)

try:
    W_opt, Y_opt = solver.solve()
    
    # Recover P and K
    P = np.linalg.inv(W_opt)
    K = Y_opt @ P
    
    print("Success! Solver finished.")
    print(f"Feedback Gain K:\n{K}")
    
except Exception as e:
    print(f"Solver crashed: {e}")


