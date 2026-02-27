#!/usr/bin/env bash
set -e

# Global cert behavior switch
if [[ "${ALLOW_INSECURE_CERTS:-false}" == "true" ]]; then
	unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
	export SSL_NO_VERIFY=1
else
	export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
	export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
	export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

# Load compass
#source load_compass_1.8.0_mpich.sh
#echo "compass conda env activated"

# Execute the command passed to the container
exec "$@"
