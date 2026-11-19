#!/usr/bin/env bash
set -e

# Always verify TLS certificates against the system CA bundle
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Load compass
#source load_compass_1.8.0_mpich.sh
#echo "compass conda env activated"

# Execute the command passed to the container
exec "$@"
