import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'V2'),str(ROOT/'V2/materiales'),str(ROOT/'validation')]
from cold_spatial import face_values,add_high_order,reconstruction_marks
from cold_spatial import spatial_context
from unittest.mock import patch
import equations
from config import FlameCase
from problem import FreeFlameProblem
from run_saved_comparison import _resolve_mechanism
from species_backend_native import NativeSpeciesBackend
from benchmark_cold_strategies import strategy_gate


class SpatialTests(unittest.TestCase):
    def test_zero_calls_or_fallback_never_certify_candidate(self):
        self.assertFalse(strategy_gate('limited-spatial',[dict(assembly_calls=0)]))
        self.assertFalse(strategy_gate('limited-spatial',[dict(assembly_calls=5,assembly_errors=['error'])]))
        self.assertTrue(strategy_gate('limited-spatial',[dict(assembly_calls=5,assembly_errors=[])]))

    def test_linear_exact_and_species_closure(self):
        z=np.linspace(0,1,31)**1.3
        y=np.array([.2+.1*z,.3-.04*z,.5-.06*z])
        for direction in (-1.,1.):
            f=face_values(z,y,np.full(z.size-1,direction),True)
            mid=.5*(z[:-1]+z[1:])
            np.testing.assert_allclose(f,np.array([.2+.1*mid,.3-.04*mid,.5-.06*mid]),atol=2e-16)
            np.testing.assert_allclose(f.sum(axis=0),1.,atol=2e-16)

    def test_bounded_faces(self):
        z=np.linspace(0,1,101)
        a=.5+.45*np.tanh(30*(z-.4))
        y=np.array([a,1-a])
        for direction in (-1.,1.):
            f=face_values(z,y,np.full(100,direction),True)
            self.assertGreaterEqual(f.min(),0.)
            self.assertLessEqual(f.max(),1.)

    def test_embedded_indicator_detects_curvature_not_linear_slope(self):
        z=np.linspace(0,1,9)
        self.assertEqual(reconstruction_marks(z,{'linear':2+3*z},1e-5),set())
        self.assertTrue(reconstruction_marks(z,{'quadratic':z*z},1e-5))

    def test_smooth_interior_second_order(self):
        errors=[]
        for n in (65,129,257):
            z=np.linspace(0,1,n)
            q=np.exp(z)[None,:]
            face=face_values(z,q,np.ones(n-1))[0]
            derivative=np.diff(face)/(z[2:]-z[:-2]) * 2
            errors.append(np.max(np.abs(derivative[2:-2]-np.exp(z[1:-1])[2:-2])))
        self.assertTrue(all(a/b>3.7 for a,b in zip(errors,errors[1:])),errors)

    def test_correction_keeps_boundary_and_algebraic_rows(self):
        n,k=9,2
        z=np.linspace(0,1,n)
        y=np.array([.3+.1*z,.7-.1*z])
        f=np.zeros(n*(k+2))
        add_high_order(f,np.ones(n),300+z,y,z,np.ones(n),np.ones(n),
                       np.tile(300+z,(k,1)),np.zeros((k,n-1)),np.ones(k),True)
        np.testing.assert_allclose(f,0.,atol=1e-12)
        self.assertEqual(reconstruction_marks(z,{'constant':np.ones(n)}),set())

    def test_real_strided_state_executes_and_jacobian_stays_low_order(self):
        p=FreeFlameProblem(FlameCase(mech=_resolve_mechanism('h2o2.yaml'),fuel='H2'))
        p.backend=NativeSpeciesBackend(p)
        p.use_numba_kinetics=True
        x=p.make_initial_guess()
        p.setup_fixed_temperature(T_profile=x.reshape(p.n_points,-1)[:,1])
        low=equations.residual(x,p)
        j0=equations._block_tridiag_jacobian_local(equations.residual,x,p)
        records=[]
        with spatial_context(records):
            high=equations.residual(x,p)
            j1=equations._block_tridiag_jacobian_local(equations.residual,x,p)
        self.assertGreater(records[0]['assembly_calls'],0)
        self.assertEqual(records[0]['assembly_errors'],[])
        self.assertGreater(np.max(abs(high-low)),1e-3)
        for attr in ('lower','diag','upper'):
            np.testing.assert_array_equal(getattr(j0,attr),getattr(j1,attr))

    def test_assembly_error_cannot_silently_revert(self):
        p=FreeFlameProblem(FlameCase(mech=_resolve_mechanism('h2o2.yaml'),fuel='H2'))
        p.backend=NativeSpeciesBackend(p)
        x=p.make_initial_guess()
        with patch('cold_spatial.add_high_order',side_effect=RuntimeError('injected')):
            with spatial_context([]),self.assertRaisesRegex(RuntimeError,'fallback forbidden'):
                equations.residual(x,p)


if __name__=='__main__':
    unittest.main()
