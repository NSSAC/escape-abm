"""Run postgres database locally."""

import time
import shlex
import shutil
import atexit
import subprocess
from random import choice
from pathlib import Path
from textwrap import dedent

import psycopg2
import netifaces

PG_HBA_DEFAULT = """
# PostgreSQL Client Authentication Configuration File
# ===================================================
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     trust
# IPv4 local connections:
host    all             all             127.0.0.1/32            trust
# IPv6 local connections:
host    all             all             ::1/128                 trust
# Allow replication connections from localhost, by a user with the replication privilege.
local   replication     all                                     trust
host    replication     all             127.0.0.1/32            trust
host    replication     all             ::1/128                 trust

# Config to be used inside a trusted cluster only
host    all             all             0.0.0.0/0               password
"""

PG_CONFIG_DEFAULT = """
# Postgresql configuration
# ========================

# Assuming Postgres is allowed to use 8 GB RAM and 2 CPU Cores

# Connections and Authentication

listen_addresses = '*'
unix_socket_directories = ''

max_connections = 100
superuser_reserved_connections = 0
max_wal_senders = 0

# Resource Consumption

shared_buffers = 2GB
work_mem = 1MB
maintenance_work_mem = 32GB

dynamic_shared_memory_type = posix

effective_io_concurrency = 10

max_worker_processes = 2
max_parallel_workers = 2
max_parallel_workers_per_gather = 1
max_parallel_maintenance_workers = 1

# Write Ahead Log

wal_level = minimal
fsync = true
wal_sync_method = fdatasync

min_wal_size = 80MB
max_wal_size = 1GB

# Query Planning
seq_page_cost = 1.0
random_page_cost = 4.0
effective_cache_size = 6GB
default_statistics_target = 1000

# Logging

log_min_messages = info
log_min_error_statement = error

log_connections = on
log_disconnections = on
log_lock_waits = on

log_timezone = 'America/New_York'

# - Locale and Formatting -

datestyle = 'iso, mdy'

timezone = 'America/New_York'

lc_messages = 'en_US.UTF-8'
lc_monetary = 'en_US.UTF-8'
lc_numeric = 'en_US.UTF-8'
lc_time = 'en_US.UTF-8'

default_text_search_config = 'pg_catalog.english'
"""


def random_str32() -> str:
    lower = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    n = 32

    return "".join(choice(lower + digits) for _ in range(n))


def wait_for_db_start(dsn: dict, server: subprocess.Popen) -> None:
    print("Waiting for database to start .", end="", flush=True)
    while True:
        if server.poll() is not None:
            raise RuntimeError("Server is not running.")

        try:
            with psycopg2.connect(**dsn):
                print("")
                return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)


def create_pg_database(
    db_dir: Path,
    initdb_exe: Path,
    username: str,
    password: str,
    verbose: bool,
):
    db_dir.mkdir(parents=True, exist_ok=True)

    (db_dir / "tmp").mkdir(parents=True, exist_ok=True)

    # Create the password file
    pass_file = db_dir / "pass.txt"
    pass_file.write_text(password)

    # Initialize the database directory
    cmd = f"""
    '{initdb_exe!s}'
        -A trust
        -D '{db_dir!s}/postgres'
        -U '{username}'
        --pwfile '{pass_file!s}'
    """
    if verbose:
        cmd_str = dedent(cmd.strip())
        print(f"executing: {cmd_str}")
    cmd = shlex.split(cmd)

    with open(db_dir / "initdb.log", "at") as fobj:
        subprocess.run(
            cmd,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=fobj,
            stderr=subprocess.STDOUT,
        )

    # Copy over the optimized postgres config
    with open(db_dir / "postgres/postgresql.conf", "wt") as fobj:
        fobj.write(PG_CONFIG_DEFAULT)

    with open(db_dir / "postgres/pg_hba.conf", "wt") as fobj:
        fobj.write(PG_HBA_DEFAULT)


class PostgresDB:
    def __init__(
        self,
        db_dir: Path | str,
        postgres_exe: Path | str | None = None,
        interface: str = "lo",
        port: int = 15432,
        username: str = "postgres",
        password: str | None = None,
        dbname: str = "default",
        delete_if_exists: bool = False,
        connect_timeout: int = 10,
        verbose: bool = False,
    ):
        db_dir = Path(db_dir)
        if password is None:
            password = random_str32()

        if postgres_exe is None:
            postgres_exe = shutil.which("postgres")
            if postgres_exe is None:
                raise RuntimeError("Unable to find postgres executable.")
        postgres_exe = Path(postgres_exe)

        initdb_exe = postgres_exe.parent / "initdb"
        createdb_exe = postgres_exe.parent / "createdb"

        self.host = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]["addr"]
        self.port = port
        self.username = username
        self.password = password
        self.dbname = dbname

        if delete_if_exists and db_dir.exists():
            shutil.rmtree(db_dir)

        run_create_db = False
        if not db_dir.exists():
            print("Initializing database directory ...")
            create_pg_database(
                db_dir,
                initdb_exe,
                username,
                password,
                verbose,
            )
            run_create_db = True

        # Start postgres in the background
        print("Starting server ...")
        cmd = f"""
        '{postgres_exe!s}'
            -p {port}
            -k '{db_dir!s}/tmp'
            -D '{db_dir!s}/postgres'
        """
        if verbose:
            cmd_str = dedent(cmd.strip())
            print(f"executing: {cmd_str}")
        cmd = shlex.split(cmd)

        with open(db_dir / "postgres.log", "at") as fobj:
            self._proc: subprocess.Popen | None = subprocess.Popen(
                cmd,
                stdout=fobj,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )

        self.dsn = dict(
            host=self.host,
            port=port,
            user=username,
            password=password,
            dbname=dbname,
            connect_timeout=connect_timeout,
        )

        self.sqlalchemy_url = (
            f"postgresql://{username}:{password}@{self.host}:{port}/{dbname}"
        )

        init_dsn = dict(
            host="localhost",
            port=port,
            user=username,
            password=password,
            dbname="template1",
            connect_timeout=connect_timeout,
        )
        wait_for_db_start(init_dsn, self._proc)

        if run_create_db:
            # Create the database
            print(f"Creating database {dbname} ...")
            cmd = f"'{createdb_exe!s}' -h localhost -p {port} -U {username} {dbname}"
            if verbose:
                cmd_str = dedent(cmd.strip())
                print(f"executing: {cmd_str}")
            cmd = shlex.split(cmd)
            with open(db_dir / "createdb.log", "at") as fobj:
                subprocess.run(
                    cmd,
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=fobj,
                    stderr=subprocess.STDOUT,
                )

        atexit.register(self.close)

    def close(self):
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exc_type, exc_val, exc_tb = exc_type, exc_val, exc_tb
        self.close()
