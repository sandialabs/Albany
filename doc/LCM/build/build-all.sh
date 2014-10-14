#!/bin/bash

source ./env-all.sh

cd "$LCM_DIR"

case "$SCRIPT_NAME" in
    build-all.sh)
	;&
    config-all.sh)
	;&
    clean-all.sh)
	;&
    test-all.sh)
	;&
    mail-all.sh)
	;&
    clean-config-all.sh)
	;&
    clean-config-build-all.sh)
	;&
    clean-config-build-test-all.sh)
	;&
    clean-config-build-test-mail-all.sh)
	;&
    config-build-all.sh)
	;&
    config-build-test-all.sh)
	;&
    config-build-test-mail-all.sh)
	;&
    build-test-all.sh)
	;&
    build-test-mail-all.sh)
	;&
    test-mail-all.sh)
	COMMAND="$LCM_DIR/${SCRIPT_NAME%-*}.sh"
	;;
    *)
	echo "Unrecognized script name"
	exit 1
	;;
esac

for PACKAGE in $PACKAGES; do
    for TOOL_CHAIN in $TOOL_CHAINS; do
	for BUILD_TYPE in $BUILD_TYPES; do
	    "$COMMAND" "$PACKAGE" "$TOOL_CHAIN" "$BUILD_TYPE" "$NUM_PROCS"
	done
    done
done

cd "$LCM_DIR"
