Howto setup Jupyter Lab for local EpiSim37 development
======================================================

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

  $ conda create -n episim37 python=3.11 nodejs=20 jupyterlab=4.1.8 gxx_impl_linux-64 cmake ninja hdf5
  $ conda activate episim37

Install jupyterlab_esl37 (for ESL37 file type support),
episim37 (for ESL37 language server),
and jupyterlab-lsp (for allowing Jupyter Lab to use the language server).


.. code::

  $ pip install jupyterlab_esl37 episim37 jupyterlab-lsp

Configure jupyterlab-lsp.

.. code::

  # Locate episim37 executable
  $ which episim37
  /path/to/miniconda3/envs/episim37/bin/episim37

  # Ensure config directory exits
  $ mkdir -p $HOME/.jupyter/jupyter_server_config.d

Create the jupyterlab-lsp config file with the following contents:

.. code::

  # $HOME/.jupyter/jupyter_server_config.d/esl37-ls.json

  {
      "LanguageServerManager": {
          "language_servers": {
              "episim37": {
                  "version": 2,
                  "argv": [
                      "/path/to/miniconda3/envs/episim37/bin/episim37",
                      "language-server",
                      "io-server"
                  ],
                  "languages": [
                      "esl37"
                  ],
                  "mime_types": [
                      "text/esl37"
                  ]
              }
          }
      }
  }

Replace `/path/to/miniconda3/envs/episim37/bin/episim37` with
the real path of the episim37 executable.

Jupyter Lab should now be configured for opening ESL37 files.
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

