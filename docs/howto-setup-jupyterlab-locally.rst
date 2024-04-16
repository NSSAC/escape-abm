Howto setup Jupyter Lab for local EpiSim37 development
======================================================

To run EpiSim37 on a local machine one needs
to install a C++ compiler, CMake, and HDF5 libraries
using their system's package manager.
For the rest of the dependencies we recommend
using the Conda package manager.

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

Please ensure you have the latest version of conda.

.. code::

   $ conda update -n base conda
   $ conda --version
   conda 24.3.0

Create and activate a conda environment with Python, NodeJS, and Jupyter Lab.

.. code::

  $ conda create -n episim37 python=3.11 nodejs=20 jupyterlab=4
  $ conda activate episim37

Install Jupyter Lab extension for ESL37 file type support.

.. code::

  $ pip install jupyterlab_esl37

Install episim37.

.. code::

  $ pip install episim37

Install jupyterlab-lsp for ESL37 language server support inside Jupyter Lab.

.. code::

  $ pip install jupyterlab-lsp

Configure jupyterlab-lsp so that it find ESL37 language server.

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

   $ jupyter lab

