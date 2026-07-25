# SPDX-License-Identifier: Apache-2.0
"""Guards against the deprecated ``pltpu.*`` spellings in the collectives kernels.

JAX moved ``semaphore``/``semaphore_read``/``semaphore_signal``/``semaphore_wait``/``DeviceIdType``
and ``HOST`` from ``pltpu`` to ``pl``.  The old names survive only as ``__getattr__`` shims
(``jax/experimental/pallas/tpu.py``) that warn and then return the identical object, so a stale
spelling is invisible until JAX drops the shim and tracing raises ``AttributeError``.  Both checks
below are CPU-only: the shims fire at Python attribute-access time, i.e. while the kernel body is
traced, well before any Mosaic lowering.
"""

import pathlib
import re
import warnings

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

from tpu_inference.kernels import collectives
from tpu_inference.kernels.collectives import util

jax.config.parse_flags_with_absl()

# jax/experimental/pallas/tpu.py::_deprecations
_DEPRECATED_PLTPU_NAMES = (
    'DeviceIdType',
    'HOST',
    'semaphore',
    'semaphore_read',
    'semaphore_signal',
    'semaphore_wait',
)
# \b so that plain `semaphore` does not also match `semaphore_signal` and triple-report it.
_DEPRECATED_PLTPU_RE = re.compile(r'\bpltpu\.(%s)\b' %
                                  '|'.join(_DEPRECATED_PLTPU_NAMES))


class CollectivesDeprecatedApiTest(jtu.JaxTestCase):

    def test_local_barrier_uses_no_deprecated_pltpu_api(self):

        def kernel(o_ref):
            util.local_barrier(0, 1, double_barrier=True)
            o_ref[...] = jnp.zeros_like(o_ref)

        f = pl.pallas_call(kernel,
                           out_shape=jax.ShapeDtypeStruct((8, 128),
                                                          jnp.float32))
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # eval_shape traces the kernel body and stops there -- no device needed.
            jax.eval_shape(f)

    def test_collectives_package_free_of_deprecated_pltpu_spellings(self):
        # The trace above only reaches the code local_barrier executes; the rest of the package is
        # covered statically, which is also how the sites this test was written for were missed.
        stale = []
        for path in sorted(
                pathlib.Path(collectives.__file__).parent.glob('*.py')):
            stale += [
                f'{path.name}:{m.group(0)}'
                for m in _DEPRECATED_PLTPU_RE.finditer(path.read_text())
            ]
        self.assertEmpty(stale)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())
