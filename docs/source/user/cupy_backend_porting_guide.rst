Cupy backend porting guide
==========================

If you have a GPU, also install the python packages `cupy` and optionally
`pyvkfft`:

.. code-block:: shell

  pip install cupy-cuda12x pyvkfft

or

.. code-block:: shell

  pip install cupy-cuda11x pyvkfft

depending on your cuda version (check with `nvidia-smi`).

The most important change in the interface is that `ift.Field.val` is not a
`np.ndarray` anymore but rather a so-called `ift.AnyField`. If you actually need
a `np.ndarray` (e.g., for plotting), change `.val` to `.asnumpy()`. In most
cases, you should not call `.asnumpy()` in `Operator.apply` because `.asnumpy()`
triggers a copy from GPU to host.

Luckily, `ift.Field.val` almost identically behaves to a `np.ndarray`, so
hopefully you should not need to change much in `Operator.apply()`.

If you have self-written operators that need some static array data (think of
the diagonal of an `ift.DiagonalOperator`), make sure to wrap these arrays as
`ift.AnyArray` and copy them to the correct device during `Operator.apply`. For
reference see, e.g., `ift.MaskOperator`.

At the very end, you can call, e.g., `ift.optimize_kl(..., device_id=0)` to
initialize your Fields in latent space on the GPU.

Debugging tools
---------------

In the next step, to make sure that everything actually runs on the GPU, you can
enable debug logging which tells you about every CPU <-> GPU copy.

.. code-block:: python

  import logging
  import nifty.cl as ift

  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ift.logger.addHandler(ch)

The second important debugging tool is `ift.check_operator` and
`ift.check_linear_operator`. Additionally, to everything else it makes sure that
input and output live on the same device.

The last debugging helpers are:

.. code-block:: python

  import nifty.cl as ift

  ift.config.update("fail_on_device_copy", True)

With this, the code crashes when an `ift.AnyArray` is copied across devices.
This is helpful to identify unintended copies.

There is also:

.. code-block:: python

  import nifty.cl as ift

  ift.config.update("break_on_device_copy", True)
  ift.config.update("fail_on_nontrivial_anyarray_creation_on_host", True)
