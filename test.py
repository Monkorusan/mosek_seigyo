import mosek.fusion as mf
from abc import ABC, abstractmethod
import numpy as np
from Mathwrapper import Mathwrapper as mw
from typing import Any

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
        n = self.A_np.shape[0]
        self._setup_geometry(n)
        
    def solve(self):
        """without the Mathwrapper, def solve would look something like this:
            with mf.Model(self.name) as M:
                P = M.variable("P", self.psd_domain)
                A = mf.Matrix.dense(self.A_np)
                PA = self.mul(P, A)
                Q = mf.Expr.add(PA, mf.Expr.transpose(PA))
                M.constraint(mf.Expr.sub(P, self.I), self.psd_domain)
                M.constraint(mf.Expr.sub(Q, self.I), self.psd_domain)
                M.solve()
                status = M.getProblemStatus()
                return status == mf.ProblemStatus.PrimalAndDualFeasible"""
        with mf.Model(self.name) as M:
            P = mw(M.variable("P", self.psd_domain))
            A = mw(mf.Matrix.dense(self.A_np))
            I = mw(self.I)
            Q = P@A + A.T@P
            cnstr1 = P - I #P-I is pos semi def \equiv P is pos def
            cnstr2 = Q - I
            # M.constraint(cnstr1.Expr, self.psd_domain)
            M.constraint((P>>I).Expr, self.psd_domain)
            M.constraint((Q>>I).Expr, self.psd_domain)
            # M.constraint(cnstr2.Expr, self.psd_domain)
            M.solve()
            status = M.getProblemStatus()
            if status == mf.ProblemStatus.PrimalAndDualFeasible:
                return P.val()

A = np.array([
    [ 2 ,-1 , 0],
    [-1 , 2 ,-1],
    [ 0 ,-1 , 2]])
solver = LyapunovIneqSolver(A)
print(solver.solve())


