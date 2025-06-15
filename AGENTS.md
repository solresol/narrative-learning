This repository includes scripts that require PostgreSQL. The `envsetup.sh` script installs both the `postgresql` server and client packages so the command line tools are available. Testing or running utilities may assume that PostgreSQL is present on the system.

Python commands are typically executed with the `uv` package manager. The setup script ensures `uv` is installed so `uv run` can be used for project scripts.
