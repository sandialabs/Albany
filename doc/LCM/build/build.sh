#!/bin/bash

source ./env-single.sh

cd "$LCM_DIR"

case "$SCRIPT_NAME" in
    build.sh)
	;;
    config.sh)
	;;
    clean.sh)
	;;
    test.sh)
	;;
    mail.sh)
	;;
    clean-config.sh)
	;;
    clean-config-build.sh)
	;;
    clean-config-build-test.sh)
	;;
    clean-config-build-test-mail.sh)
	;;
    config-build.sh)
	;;
    config-build-test.sh)
	;;
    config-build-test-mail.sh)
	;;
    build-test.sh)
	;;
    build-test-mail.sh)
	;;
    test-mail.sh)
	;;
    *)
	echo "Unrecognized script name"
	exit 1
	;;
esac

echo "------------------------------------------------------------"
echo -e "$PACKAGE_NAME directory\t: $PACKAGE_DIR"
echo -e "Install directory \t: $INSTALL_DIR"
echo -e "Build directory\t\t: $BUILD_DIR"
echo "------------------------------------------------------------"

case "$SCRIPT_NAME" in
    *clean*)
	echo "CLEANING UP $PACKAGE_STRING ..."
	echo "------------------------------------------------------------"
	# Remove install directory for Trilinos only.
	case "$PACKAGE" in
	    trilinos)
		if [ -d "$INSTALL_DIR" ]; then
		    rm "$INSTALL_DIR" -rf
		fi
		;;
	    albany)
		;;
	    *)
		echo "Unrecognized package option"
		exit 1
		;;
	esac
	if [ -d "$BUILD_DIR" ]; then
	    rm "$BUILD_DIR" -rf
	fi
	;;&
    *config*)
	echo "CONFIGURING $PACKAGE_STRING ..."
	echo "------------------------------------------------------------"
	if [ ! -d "$BUILD_DIR" ]; then
	    mkdir "$BUILD_DIR"
	fi
	if [ -f "$BUILD_DIR/$CONFIG_FILE" ]; then
	    rm "$BUILD_DIR/$CONFIG_FILE" -f
	fi
	cp -p "$CONFIG_FILE" "$BUILD_DIR"
	cd "$BUILD_DIR"
	sed -i -e "s|ompi_cc|$OMPI_CC|g;" "$CONFIG_FILE"
	sed -i -e "s|ompi_cxx|$OMPI_CXX|g;" "$CONFIG_FILE"
	sed -i -e "s|ompi_fc|$OMPI_FC|g;" "$CONFIG_FILE"
	sed -i -e "s|install_dir|$INSTALL_DIR|g;" "$CONFIG_FILE"
	sed -i -e "s|build_type|$BUILD_STRING|g;" "$CONFIG_FILE"
	sed -i -e "s|cmake_cxx_flags|$CMAKE_CXX_FLAGS|g;" "$CONFIG_FILE"
	sed -i -e "s|package_dir|$PACKAGE_DIR|g;" "$CONFIG_FILE"
	./"$CONFIG_FILE"
	echo "------------------------------------------------------------"
	;;&
    *build*)
	if [ ! -d "$BUILD_DIR" ]; then
	    echo "Build directory does not exist. Run:"
	    echo "  [clean-]config-build.sh $1 $2 $3 $4"
	    echo "to create."
	    exit 1
	fi
	cd "$BUILD_DIR"
	echo "REBUILDING $PACKAGE_STRING ..."
	echo "------------------------------------------------------------"
	cd "$BUILD_DIR"
	echo "WARNINGS AND ERRORS REDIRECTED TO $ERROR_LOG"
	echo "------------------------------------------------------------"
	if [ -f "$ERROR_LOG" ]; then
	    rm "$ERROR_LOG"
	fi

	case "$PACKAGE" in
	    trilinos)
		make -j $NUM_PROCS 2> "$ERROR_LOG"
		STATUS=$?
		if [ $STATUS -ne 0 ]; then
		    echo "*** MAKE COMMAND FAILED ***"
		else
		    make install
		    STATUS=$?
		    if [ $STATUS -ne 0 ]; then
			echo "*** MAKE INSTALL COMMAND FAILED ***"
		    fi
		fi
		;;
	    albany)
		make -j $NUM_PROCS 2> "$ERROR_LOG"
		STATUS=$?
		if [ $STATUS -ne 0 ]; then
		    echo "*** MAKE COMMAND FAILED ***"
		fi
		;;
	    *)
		echo "Unrecognized package option"
		exit 1
		;;
	esac
	;;&
    *test*)
	#No Trilinos testing
	case "$PACKAGE" in
	    trilinos)
		;;
	    albany)
		if [ ! -d "$BUILD_DIR" ]; then
		    echo "Build directory does not exist. Run:"
		    echo "  [clean-]config-build.sh $1 $2 $3 $4"
		    echo "to create."
		    exit 1
		fi
		cd "$BUILD_DIR"
		echo "TESTING $PACKAGE_STRING ..."
		echo "------------------------------------------------------------"
		ctest --timeout 300 . | tee "$TEST_LOG"
		;;
	    *)
		echo "Unrecognized package option"
		exit 1
		;;
	esac
	;;&
    *mail*)
	if [ -f "$TEST_LOG" ]; then
	    SUCCESS_RATE=`grep "tests failed" "$TEST_LOG"`
	    HEADER="LCM TESTS: $HOST, $TOOL_CHAIN $BUILD_TYPE, $SUCCESS_RATE"
	    mail -r "$FROM" -s "$HEADER" "$TO" < "$TEST_LOG"
	fi
	;;&
esac

cd "$LCM_DIR"
