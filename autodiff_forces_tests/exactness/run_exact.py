"""Runs the exactness tests of test_autodiff_forces.py without pytest (both systems, both rebalancing rules)."""
import os, sys, types, traceback
sys.path.insert(0, os.environ["VAFPY_SRC"])
import test_autodiff_forces as t

npass = nfail = 0
for name in t.SYSTEMS:
    system = t.system(types.SimpleNamespace(param=name))
    cases = [(f, ()) for f in (t.test_expm_taylor_matches_scipy, t.test_step_matches_propagate_walkers,
                               t.test_trial_walkers_give_hf_derivative,
                               t.test_forward_derivative_equals_finite_difference_without_rebalancing)]
    cases += [(t.test_forward_equals_reverse, (m,)) for m in ("martin", "zero")]
    for f, extra in cases:
        try:
            f(system, *extra)
            npass += 1
            print("PASS", name, f.__name__, *extra, flush=True)
        except Exception:
            nfail += 1
            print("FAIL", name, f.__name__, *extra, flush=True)
            traceback.print_exc()
print(f"{npass} passed, {nfail} failed")
