PyTDA README
------------

This software provides Python functions that will estimate turbulence from
Doppler radar data. It is tested and working under Python 2.7 and 3.4.

For help see `HELP` file. For license see `LICENSE.md`.


Installation
------------

Make sure this directory is in your `PYTHONPATH`.

Install [Py-ART](https://github.com/ARM-DOE/pyart).

Run `compile_pytda_cython_code.sh` from the command line. The shared object file
that is created by this is only valid for the version of Python that it was compiled under.


Using PyTDA
-----------
```
import pytda
```

PyTDA (among other modules) is discussed in this [conference presentation]
(https://ams.confex.com/ams/95Annual/webprogram/Paper262779.html)

See the notebooks directory for a demonstration Jupyter notebook.
