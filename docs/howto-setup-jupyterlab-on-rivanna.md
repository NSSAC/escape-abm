# Howto setup Jupyter Lab for ESCAPE development on Rivanna


## Setup slurm-hpc-workflows

Please follow the instructions to setup slrum-hpc-workflows on Rivanna.
As part of the process you will install Jupyter Lab for which we can install extensions.

* [How to setup slurm-hpc-workflows on Rivanna](https://github.com/parantapa/slurm-hpc-workflows/blob/main/docs/rivanna-setup.md)

## Install ESCAPE and Jupyter Lab extensions


If you already have a jupyter lab session running (check using `squeue` command)
you need to first stop it (using `scancel`).
Please ensure that you do not have a jupyter lab session running before proceeding.

We will install jupyterlab_esl (for ESL file type support),
escape_abm (for ESL language server),
and jupyterlab-lsp (for allowing Jupyter Lab to use the language server).

Ensure you are logged into Rivanna via ssh. Also ensure that myenv is still activated.

```sh
conda install 
pip install jupyterlab_esl escape_abm jupyterlab-lsp "pygls>=1.1.1,<2.0.0" cmake ninja hdf5
```

## Configure jupyterlab-lsp

Note the location of the ESCAPE command line frontend

```sh
which esc
```
```
/path/to/envs/escape/bin/esc
```

Create jupyterlab-lsp config file for ESL.
Ensure config directory exits.

```sh
mkdir -p $HOME/.jupyter/jupyter_server_config.d
```

Create `$HOME/.jupyter/jupyter_server_config.d/esl-ls.json` with the following contents

```json
{
    "LanguageServerManager": {
        "language_servers": {
            "escape": {
                "version": 2,
                "argv": [
                    "/path/to/envs/escape/bin/esc",
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
```

Replace `/path/to/envs/escape/bin/esc` with
the real path of the `esc` executable.

Jupyter Lab should now be configured for opening ESL files.

## Start Jupyter from slurm-hpc-workflows

First start [Jupyter Lab from slurm-hpc-workflows](https://github.com/parantapa/slurm-hpc-workflows/blob/main/docs/rivanna-setup.md#start-jupyter-from-slurm-workflows-on-rivanna).

Next a socks proxy over ssh is not already running
[start the socks proxy](https://github.com/parantapa/slurm-hpc-workflows/blob/main/docs/rivanna-setup.md#setup-a-socks-proxy-over-ssh-on-your-local-computer-to-enable-your-browser-to-connect-to-rivanna).

You should now be able to [connect to the Jupyter lab on Rivanna](https://github.com/parantapa/slurm-hpc-workflows/blob/main/docs/rivanna-setup.md#setup-a-socks-proxy-over-ssh-on-your-local-computer-to-enable-your-browser-to-connect-to-rivanna)

## Next Steps

Check out the `run-example1.ipynb` notebook in the `examples` directory
in the newly opened jupyter lab to see how to use ESCAPE.
