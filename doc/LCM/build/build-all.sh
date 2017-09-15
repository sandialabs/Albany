#!/bin/bash

update_wiki () {
    echo "IN UPDATE WIKI"
    cd "$LCM_DIR"
    STATUS_LOG="$PACKAGE-$ARCH-$TOOL_CHAIN-$BUILD_TYPE-status.log"
    echo "STATUS_LOG=$STATUS_LOG"
    if [[ -f "$STATUS_LOG" && -d "$LCM_DIR/Albany.wiki" ]]; then
        echo "IN IF UPDATE"
	SRC="$LCM_DIR/Albany/doc/LCM/test/$WIKI_TEMPLATE"
	DEST="$LCM_DIR/Albany.wiki/$WIKI_TEMPLATE"
	cd "$LCM_DIR/Albany.wiki"
        git pull
	cp -p "$SRC" "$DEST"
	cd "$LCM_DIR/Trilinos"
	TRILINOS_TAG=`git rev-parse HEAD`
	sed -i -e "s|ttag|$TRILINOS_TAG|g;" "$DEST"
	cd "$LCM_DIR/Albany"
	ALBANY_TAG=`git rev-parse HEAD`
	sed -i -e "s|atag|$ALBANY_TAG|g;" "$DEST"
	MSG="Update latest known good commits"
	cd "$LCM_DIR/Albany.wiki"
	git add "$DEST"
	git commit -m "$MSG"
	git push
	cd "$LCM_DIR"
    fi
}

source ./env-all.sh

cd "$LCM_DIR"
SCRIPT_NAME=`basename $0`

case "$SCRIPT_NAME" in
    build-all.sh)
	;&
    config-all.sh)
	;&
    clean-all.sh)
	;&
    test-all.sh)
	;&
    dash-all.sh)
	;&
    clean-config-all.sh)
	;&
    clean-config-build-all.sh)
	;&
    clean-config-build-test-all.sh)
	;&
    clean-config-build-test-dash-all.sh)
	;&
    config-build-all.sh)
	;&
    config-build-test-all.sh)
	;&
    config-build-test-dash-all.sh)
	;&
    build-test-all.sh)
	;&
    build-test-dash-all.sh)
	;&
    test-dash-all.sh)
	COMMAND="$LCM_DIR/${SCRIPT_NAME%-*}.sh"
	;;
    *)
	echo "Unrecognized script name in build-all: $SCRIPT_NAME"
	exit 1
	;;
esac
WIKI_TEMPLATE="LCM-Status:-Last-known-commits-that-work.md"

KERNEL_VERSION=`uname -r`
PLATFORM="unknown"
if [[ ${KERNEL_VERSION} == *"fc"* ]]; then
    PLATFORM="fedora"
elif [[ ${KERNEL_VERSION} == *"el"* ]]; then
    PLATFORM="sems"
elif [[ ${KERNEL_VERSION} == *"chaos"* ]]; then
    PLATFORM="cluster"
else
    echo "Unrecongnized platform. Valid platforms: fc, el, chaos"
    uname -r
    exit 1
fi

# Use different variable names for loop counters so they do not
# conflict with the variables defined by the module command.
for P in $PACKAGES; do
    for A in $ARCHES; do
        for TC in $TOOL_CHAINS; do
            for BT in $BUILD_TYPES; do
                MODULE="$A"-"$TC"-"$BT"
                echo "MODULE: $MODULE"
		echo "PLATFORM: $PLATFORM"
                module purge
                module load lcm-"$PLATFORM"
                module load "$MODULE"
                "$COMMAND" "$P" "$NUM_PROCS"
                # Update wiki after compiling Albany with gcc release only.
                case "$P" in
                    albany)
	                case "$A" in
	                    serial)
		                case "$BT" in
		                    release)
			                case "$TC" in
			                    gcc)
                                                # PACKAGE is not set in modules
                                                PACKAGE="$P"
				                update_wiki
				                ;;
			                    *)
				                ;;
			                esac
			                ;;
		                    *)
			                ;;
		                esac
		                ;;
	                    *)
		                ;;
	                esac
	                ;;
                    *)
	                ;;
                esac
            done
        done
    done
done

cd "$LCM_DIR"
