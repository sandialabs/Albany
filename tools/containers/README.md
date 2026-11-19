# Docker/podman recipes for containers to be used for Albany CI

The subfolders in here will contain Dockerfiles for building docker/podman container images.
While the goal is to use these to run CI on github, they can also be used by external
collaborators to get a working environment for Albany.

## TLS certificates

The images always verify TLS certificates. If you work behind a proxy that intercepts TLS
(i.e., re-signs traffic with its own CA), add that CA certificate inside the running container
(`runner` has passwordless sudo):

```
sudo cp my-proxy-ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

Images are meant to be built on a network that does not intercept TLS (e.g., the GitHub runners).

## Trilinos version

The nightly images track Trilinos `develop`. The Trilinos commit an image was built from is
recorded by the Trilinos install in `$TRILINOS_ROOT/TrilinosRepoVersion.txt` (the PR CI jobs print
it, and Albany's configure reports it as `Trilinos_GIT_COMMIT_ID`).
