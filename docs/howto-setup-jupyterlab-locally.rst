Howto setup Jupyter Lab for local ESCAPE development
====================================================

For this setup we shall use the Conda package manager.

Installation instructions for Miniconda can be found
`here <https://docs.conda.io/en/latest/miniconda.html>`_.

Once setup is done, please ensure that your conda config contains the following:

.. code::

  # ~/.condarc

  channels:
    - conda-forge
    - defaults
  anaconda_upload: false
  auto_activate_base: false

Create and activate a conda environment.

.. code::

  $ conda create -n escape python=3.12 nodejs=20 jupyterlab=4 gxx_impl_linux-64 cmake ninja hdf5
  $ conda activate escape

Install jupyterlab_esl (for ESL file type support),
escape_abm (for ESL language server),
and jupyterlab-lsp (for allowing Jupyter Lab to use the language server).


.. code::

  $ pip install jupyterlab_esl escape_abm jupyterlab-lsp

Configure jupyterlab-lsp.

.. code::

  # Locate ESCAPE command line frontend
  $ which esc
  /path/to/miniconda3/envs/escape/bin/esc

  # Ensure config directory exits
  $ mkdir -p $HOME/.jupyter/jupyter_server_config.d

Create the jupyterlab-lsp config file with the following contents:

.. code::

  # $HOME/.jupyter/jupyter_server_config.d/esl-ls.json

  {
      "LanguageServerManager": {
          "language_servers": {
              "escape": {
                  "version": 2,
                  "argv": [
                      "/path/to/miniconda3/envs/escape/bin/esc",
                      "language-server",
                      "io-server"
                  ],
                  "languages": [
                      "esl"
                  ],
                  "mime_types": [
                      "text/esl"
                  ]
              }
          }
      }
  }

Replace `/path/to/miniconda3/envs/escape/bin/esc` with
the real path of the `esc` executable.

Jupyter Lab should now be configured for opening ESL files.
Start Jupyter Lab using the following command:

.. code::

   $ cd $HOME
   $ jupyter lab


Known Issues
------------

If your Jupyter kernel keeps crashing on a Apple Mac
it maybe due to incompatibility with `polars'.
You can try using the long term release version of polars.

.. code::

   $ pip uninstall polars
   $ pip install polars-lts-cpu

