#!/bin/bash

source ./env-all.sh

cd "$TEST_DIR"

# Clone package repositories.
for PACKAGE in $PACKAGES; do
    case "$PACKAGE" in
	trilinos)
	    PACKAGE_NAME="Trilinos"
	    REPO="git@github.com:trilinos/Trilinos.git"
            BRANCH="develop"
	    ;;
	albany)
	    PACKAGE_NAME="Albany"
	    REPO="git@github.com:SNLComputation/Albany.git"
            BRANCH="master"
	    ;;
	*)
	    echo "Unrecognized package option"
	    exit 1
	    ;;
    esac
    PACKAGE_DIR="$TEST_DIR/$PACKAGE_NAME"
    CHECKOUT_LOG="$PACKAGE-checkout.log"
    if [ -d "$PACKAGE_DIR" ]; then
	rm "$PACKAGE_DIR" -rf
    fi
    git clone -v -b "$BRANCH" "$REPO" "$PACKAGE_NAME" &> "$CHECKOUT_LOG"
done

# For now assume that if there is a DTK directory in the main LCM
# directory, it contains a DTK version that we can use for
# Trilinos.
if [ -e DataTransferKit ]; then
    cp -p -r DataTransferKit Trilinos
fi

# Clone wiki too to update info for latest known good commits.
if [ -d "Albany.wiki" ]; then
    rm "Albany.wiki" -rf
fi
git clone git@github.com:SNLComputation/Albany.wiki.git

./clean-config-build-test-dash-all.sh

cd "$TEST_DIR"
