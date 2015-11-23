#!/bin/bash

source ./env-single.sh

cd "$LCM_DIR"

if [ -f "$STATUS_LOG" ]; then
    rm "$STATUS_LOG" -f
fi
		
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

LINE="------------------------------------------------------------"

echo "$LINE"
echo -e "$PACKAGE_NAME directory\t: $PACKAGE_DIR"
echo -e "Install directory \t: $INSTALL_DIR"
echo -e "Build directory\t\t: $BUILD_DIR"
echo "$LINE"

case "$SCRIPT_NAME" in
    *clean*)
	echo "CLEANING UP $PACKAGE_STRING ..."
	echo "$LINE"
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
	echo "$LINE"
	if [ ! -d "$BUILD_DIR" ]; then
	    mkdir "$BUILD_DIR"
	fi
	if [ -f "$BUILD_DIR/$CONFIG_FILE" ]; then
	    rm "$BUILD_DIR/$CONFIG_FILE" -f
	fi
	cp -p "$CONFIG_FILE" "$BUILD_DIR"
	cd "$BUILD_DIR"
	sed -i -e "s|lcm_ompi_cc|$OMPI_CC|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_ompi_cxx|$OMPI_CXX|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_ompi_fc|$OMPI_FC|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_install_dir|$INSTALL_DIR|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_build_type|$BUILD_STRING|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_package_dir|$PACKAGE_DIR|g;" "$CONFIG_FILE"
	case "$BUILD_TYPE" in
	    debug)
		sed -i -e "s|lcm_fpe_switch|ON|g;" "$CONFIG_FILE"
		;;
	    release)
		sed -i -e "s|lcm_fpe_switch|OFF|g;" "$CONFIG_FILE"
		;;
	    profile)
		sed -i -e "s|lcm_fpe_switch|OFF|g;" "$CONFIG_FILE"
		;;
	    small)
		sed -i -e "s|lcm_fpe_switch|OFF|g;" "$CONFIG_FILE"
		;;
	    *)
		echo "Unrecognized build type option"
		exit 1
		;;
	esac
	case "$ARCH" in
	    serial)
		sed -i -e "s|lcm_enable_cuda|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_uvm|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_examples|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_openmp|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_pthreads|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_cusparse|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_phalanx_index_type|INT|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_kokkos_device|SERIAL|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_tpetra_inst_pthread|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_hwloc|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_devel|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_slfad|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_slfad_size||g;" "$CONFIG_FILE"
		;;
	    openmp)
		sed -i -e "s|lcm_enable_cuda|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_uvm|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_examples|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_openmp|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_pthreads|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_cusparse|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_phalanx_index_type|INT|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_kokkos_device|OPENMP|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_tpetra_inst_pthread|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_hwloc|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_devel|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_slfad|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_slfad_size||g;" "$CONFIG_FILE"
		;;
	    pthreads)
		sed -i -e "s|lcm_enable_cuda|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_uvm|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_examples|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_openmp|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_pthreads|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_cusparse|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_phalanx_index_type|INT|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_kokkos_device|THREAD|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_tpetra_inst_pthread|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_hwloc|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_devel|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_slfad|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_slfad_size||g;" "$CONFIG_FILE"
		;;
	    cuda)
		sed -i -e "s|lcm_enable_cuda|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_uvm|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_examples|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_openmp|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_pthreads|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_cusparse|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_phalanx_index_type|UINT|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_kokkos_device|CUDA|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_tpetra_inst_pthread|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_hwloc|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_devel|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_slfad|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_slfad_size|-D SLFAD_SIZE=48|g;" "$CONFIG_FILE"
		;;
	    *)
		echo "Unrecognized architecture option"
		exit 1
		;;
	esac
	./"$CONFIG_FILE"
	echo "$LINE"
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
	echo "$LINE"
	cd "$BUILD_DIR"
	echo "WARNINGS AND ERRORS REDIRECTED TO $ERROR_LOG"
	echo "$LINE"
	if [ -f "$ERROR_LOG" ]; then
	    rm "$ERROR_LOG"
	fi

	case "$PACKAGE" in
	    trilinos)
		make -j $NUM_PROCS 2> "$ERROR_LOG"
		STATUS=$?
		if [ $STATUS -ne 0 ]; then
		    echo "*** MAKE COMMAND FAILED ***"
		    exit 1
		else
		    make install
		    STATUS=$?
		    if [ $STATUS -ne 0 ]; then
			echo "*** MAKE INSTALL COMMAND FAILED ***"
			exit 1
		    fi
                    ln -sf "$INSTALL_DIR/include" "$INSTALL_DIR/inc"
		    echo SUCCESS > "$STATUS_LOG" 
		fi
		;;
	    albany)
		make -j $NUM_PROCS 2> "$ERROR_LOG"
		STATUS=$?
		if [ $STATUS -ne 0 ]; then
		    echo "*** MAKE COMMAND FAILED ***"
		    exit 1
		fi
		echo SUCCESS > "$STATUS_LOG" 
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
		echo "$LINE"
		ctest --timeout 600 . | tee "$TEST_LOG"
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
	    BUILD_ENV="$HOST, $ARCH_NAME $TOOL_CHAIN $BUILD_TYPE"
	    HEADER="LCM TESTS: $BUILD_ENV, $SUCCESS_RATE"
	    mail -r "$FROM" -s "$HEADER" "$TO" < "$TEST_LOG"
	    STATUS=$?
	    if [ $STATUS -ne 0 ]; then
		echo "*** MAIL COMMAND FAILED ***"
		exit 1
	    fi
	fi
	;;&
esac

cd "$LCM_DIR"
