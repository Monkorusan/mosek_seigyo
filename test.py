import mosek.fusion as mf
from abc import ABC, abstractmethod
import numpy as np
from MathWrapper import MathWrapper as mw
from typing import Any
from ModelWrapper import ModelWrapper

def block(rows):
    """
    imitate np.block() method, e.g. block([[A,B],[C,D]]) then return block matrix 
    """
    raw_rows = []
    for row in rows:
        raw_row = [item.Expr if isinstance(item,MathWrapper) else item for item in rows]
        raw_rows.append(mf.Expr.hstack(raw_row)) #stack horizontally then
    return MathWrapper(mf.Expr.vstack(raw_rows)) #stack vertically then get block matrix

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
            A = M.matrix(self.A_np)
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
        self._setup_geometry(self.n)

    def solve(self):
        with ModelWrapper(self.name) as M:
            I = mw.eye(self.n)
            W = M.variable("W", self.psd_domain) # W=inv(P)
            Y = M.variable("Y", self.psd_domain) # Y=KW=K@inv(P)
            A = M.matrix(self.A_np)
            B = M.matrix(self.B_np) #todo, change modelwrapper.matrix name to const_matrix for readability
            Q = M.matrix(self.Q_np)
            R = M.matrix(self.R_np)
            
            Q_sqrt, R_sqrt = mw.sqrt(Q) , mw.sqrt(R)
            _lmi = block([[    W     , W@A.T+Y.T@B.T , W@Q_sqrt, Y.T@R_sqrt],
                          [A@W + B@Y ,      W        ,  mw.zeros(A.shape)   ,   0],
                          [ Q_sqrt@W , 0            ,         I   ,   0],
                          [ R_sqrt@Y , 0             ,         0   ,   I]])
            #todo: 0 arent 0, they are actually zero matrices.
            I = mw.eye(self.n) 


A = np.array([
    [ 2 ,-1 , 0],
    [-1 , 2 ,-1],
    [ 0 ,-1 , 2]])
solver = LyapunovIneqSolver(A)
print(solver.solve())


