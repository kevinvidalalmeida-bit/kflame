"""Experimental limited face reconstruction with the original quasi-Newton J.

Five-point nonlinear convection; the old three-point Jacobian is only a
preconditioning approximation. No claim of an exact higher-order Jacobian.
The optional reconstruction indicator is not a certified goal-error estimator.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from numba import njit
import equations
import solver


@njit(cache=True)
def slopes(z, q, shared=False):
    kmax, n = q.shape
    s = np.zeros_like(q)
    s[:, 0] = (q[:,1]-q[:,0])/(z[1]-z[0])
    s[:, -1] = (q[:,-1]-q[:,-2])/(z[-1]-z[-2])
    for j in range(1,n-1):
        hm, hp = z[j]-z[j-1], z[j+1]-z[j]
        common = 1.
        for k in range(kmax):
            left, right = (q[k,j]-q[k,j-1])/hm, (q[k,j+1]-q[k,j])/hp
            center = (hp*left+hm*right)/(hm+hp)
            s[k,j] = center
            theta = 1.
            if center != 0.:
                theta = max(0., min(1.,2.*left/center,2.*right/center))
            common = min(common,theta)
            if not shared:
                s[k,j] *= theta
        if shared:
            s[:,j] *= common
    return s


@njit(cache=True)
def face_values(z, q, mass, shared=False):
    s = slopes(z,q,shared)
    faces = np.empty((q.shape[0], z.size-1))
    for f in range(z.size-1):
        dz = .5*(z[f+1]-z[f])
        for k in range(q.shape[0]):
            faces[k,f] = q[k,f]+dz*s[k,f] if mass[f]>=0 else q[k,f+1]-dz*s[k,f+1]
    return faces


@njit(cache=True)
def add_high_order(F,u,T,Y,z,rho,cp,h,flux,invW,energy):
    n, nv = z.size,Y.shape[0]+2
    mass = .5*(rho[:-1]*u[:-1]+rho[1:]*u[1:])
    yf = face_values(z,Y,mass,True)
    tf = face_values(z,np.ascontiguousarray(T).reshape(1,n),mass)
    for j in range(1,n-1):
        hm,hp = z[j]-z[j-1],z[j+1]-z[j]
        vol=.5*(hm+hp)
        loc = j if u[j]>0 else j+1
        dz = z[loc]-z[loc-1]
        for k in range(Y.shape[0]):
            old = u[j]*(Y[k,loc]-Y[k,loc-1])/dz
            new = (mass[j]*(yf[k,j]-Y[k,j])-mass[j-1]*(yf[k,j-1]-Y[k,j]))/(rho[j]*vol)
            F[j*nv+2+k] += old-new
        if energy:
            old = u[j]*(T[loc]-T[loc-1])/dz
            new = (mass[j]*(tf[0,j]-T[j])-mass[j-1]*(tf[0,j-1]-T[j]))/(rho[j]*vol)
            F[j*nv+1] += old-new
            for k in range(Y.shape[0]):
                dh_old = (h[k,loc]-h[k,loc-1])/dz
                dh_new = (hp*(h[k,j]-h[k,j-1])/hm+hm*(h[k,j+1]-h[k,j])/hp)/(hm+hp)
                F[j*nv+1] += .5*(flux[k,j-1]+flux[k,j])*invW[k]*(dh_old-dh_new)/(rho[j]*cp[j])


def reconstruction_marks(z, profiles, tolerance=.01):
    """Embedded linear/quadratic face discrepancy, normalized by profile range.

    A refinement monitor ONLY, not an upper bound on solution error. Actual
    accuracy must be checked against an independently refined solution.
    """
    marks=set()
    for values in profiles.values():
        q=np.asarray(values)
        scale=np.ptp(q)
        if scale <= max(1e-12,.01*np.max(np.abs(q))):
            continue
        for j in range(1,len(z)-1):
            hm,hp=z[j]-z[j-1],z[j+1]-z[j]
            left,right=(q[j]-q[j-1])/hm,(q[j+1]-q[j])/hp
            # q'' from the local quadratic interpolant, independent of the
            # lower-order limiter. Taylor remainder at each adjacent face.
            curvature=2.*(right-left)/(hm+hp)
            if abs(curvature)*hm*hm/8./scale > tolerance:
                marks.add(j-1)
            if abs(curvature)*hp*hp/8./scale > tolerance:
                marks.add(j)
    return marks


@contextmanager
def spatial_context(records, adapt=False):
    original=equations._assemble_residual_numba_core
    original_residual=equations.residual
    original_jacobian=equations._block_tridiag_jacobian_local
    original_refresh=equations.refresh_block_tridiag_jacobian_columns
    analyze=solver.AdaptiveRefiner.analyze
    counts=dict(kind='limited_spatial',assembly_calls=0,refinement_calls=0,extra_marks=0,assembly_errors=[],
                jacobian='original upwind quasi-Newton',indicator='linear/quadratic face discrepancy' if adapt else None)
    def assembly(*args):
        try:
            original(*args)
            F,u,T,Y,z,rho,cp,omega,h,lam,flux,invW=args[:12]
            add_high_order(F,u,T,Y,z,rho,cp,h,flux,invW,args[15])
        except Exception as exc:
            counts['assembly_errors'].append(repr(exc))
            raise
        counts['assembly_calls']+=1
    def checked_residual(*args,**kwargs):
        result=original_residual(*args,**kwargs)
        if counts['assembly_errors']:
            raise RuntimeError('Spatial assembly fallback forbidden: '+counts['assembly_errors'][-1])
        return result
    def low_jacobian(*args,**kwargs):
        # Both f0 and all perturbed rows MUST use the lower-order residual;
        # subtracting a high-order f0 would introduce defect/delta blow-up.
        with patch.object(equations,'_assemble_residual_numba_core',original):
            return original_jacobian(*args,**kwargs)
    def low_refresh(*args,**kwargs):
        with patch.object(equations,'_assemble_residual_numba_core',original):
            return original_refresh(*args,**kwargs)
    def refined(self,z,profiles,j_fixed=None):
        insert,remove=analyze(self,z,profiles,j_fixed)
        extra=reconstruction_marks(z,profiles)
        extra={j for j in extra if z[j+1]-z[j]>=2*self.grid_min}
        counts['refinement_calls']+=1
        counts['extra_marks']+=len(extra-insert)
        insert |= extra
        remove -= {j for f in extra for j in (f,f+1)}
        return insert,remove
    with patch.object(equations,'_assemble_residual_numba_core',assembly), \
         patch.object(equations,'residual',checked_residual), \
         patch.object(solver,'residual',checked_residual), \
         patch.object(equations,'_block_tridiag_jacobian_local',low_jacobian), \
         patch.object(equations,'refresh_block_tridiag_jacobian_columns',low_refresh), \
         patch.object(solver,'refresh_block_tridiag_jacobian_columns',low_refresh):
        try:
            if adapt:
                with patch.object(solver.AdaptiveRefiner,'analyze',refined):
                    yield
            else:
                yield
        finally:
            records.append(counts)
