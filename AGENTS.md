This repository includes scripts that require PostgreSQL. The `envsetup.sh` script installs both the `postgresql` server and client packages so the command line tools are available. Testing or running utilities may assume that PostgreSQL is present on the system.

Python commands are typically executed with the `uv` package manager. The setup script ensures `uv` is installed so `uv run` can be used for project scripts.

The setup script also restores a database called `narrative`. The service runs
locally and can be accessed using the `root` role via peer authentication. You
can inspect it with `psql -U root narrative` or connect from Python using
`psycopg2.connect(dbname='narrative', user='root')`.

When testing scripts that use libpq defaults, set the `PGUSER` environment
variable to `root` so connections are made with the correct role.
