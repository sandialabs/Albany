#!/bin/bash

cd "$LCM_DIR"

source ./env-single.sh

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
	echo "Unrecognized script name: $SCRIPT_NAME"
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
		echo "Unrecognized package option in config: $PACKAGE"
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
	case "$PACKAGE" in
	    trilinos)
	        if [ -e "$PACKAGE_DIR/DataTransferKit" ]; then
                    cp -p "$DTK_FRAG" "$BUILD_DIR"
	        fi
	        if [ -e "$PACKAGE_DIR/tempus" ]; then
                    cp -p "$TEMPUS_FRAG" "$BUILD_DIR"
	        fi
		;;
	    *)
		;;
	esac
	cd "$BUILD_DIR"
        # Add DTK fragment to Trilinos config script and disable ETI as
        # it is not supported for DTK due to incompatible Global Index types.
	case "$PACKAGE" in
	    trilinos)
                # First build extra repos string
                ER=""
	        if [ -e "$PACKAGE_DIR/DataTransferKit" ]; then
                    if [ -z $ER ]; then
                        ER="DataTransferKit"
                    else
                        ER="$ER,DataTransferKit"
                    fi
                fi
                if [ ! -z $ER ]; then
                    TER=" -D Trilinos_EXTRA_REPOSITORIES:STRING=\"$ER\" \\"
                    sed -i -e "/lcm_package_dir/d" "$CONFIG_FILE"
                    echo "\\" >> "$CONFIG_FILE"
                    echo "$TER" >> "$CONFIG_FILE"
                fi

	        if [ -e "$PACKAGE_DIR/DataTransferKit" ]; then
                    TMP_FILE="/tmp/_TMP_FILE_"
                    ETION="Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
                    ETIOFF="Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=OFF"
                    cat "$CONFIG_FILE" "$DTK_FRAG" > "$TMP_FILE"
                    mv "$TMP_FILE" "$CONFIG_FILE"
                    chmod 0755 "$CONFIG_FILE"
                    sed -i -e "s|$ETION|$ETIOFF|g;" "$CONFIG_FILE"
	        fi
                if [ ! -z $ER ]; then
                    echo "lcm_package_dir" >> "$CONFIG_FILE"
                fi
		;;
	    *)
		;;
	esac
	sed -i -e "s|lcm_ompi_cc|$OMPI_CC|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_ompi_cxx|$OMPI_CXX|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_ompi_fc|$OMPI_FC|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_install_dir|$INSTALL_DIR|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_build_type|$BUILD_STRING|g;" "$CONFIG_FILE"
	sed -i -e "s|lcm_package_dir|$PACKAGE_DIR|g;" "$CONFIG_FILE"
        # Check if a custom build netcdf with pnetcdf exists and use that
        # instead of the system one to avoid failing horrible tests that
        # need this (yuck!).
        if [ ! -n "$NETCDF_INC" ]; then
            if [ -e "/usr/local/netcdf/lib/libnetcdf.so" ]; then
                NETCDF_INC=/usr/local/netcdf/include
                NETCDF_LIB=/usr/local/netcdf/lib
            else
                NETCDF_INC=/usr/include/openmpi-x86_64
                NETCDF_LIB=/usr/lib64/openmpi/lib
            fi
        fi
        sed -i -e "s|lcm_netcdf_inc|$NETCDF_INC|g;" "$CONFIG_FILE"
        sed -i -e "s|lcm_netcdf_lib|$NETCDF_LIB|g;" "$CONFIG_FILE"
	case "$BUILD_TYPE" in
	    debug)
		sed -i -e "s|lcm_fpe_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_denormal_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_cxx_flags|-msse3|g;" "$CONFIG_FILE"
		;;
	    release)
		sed -i -e "s|lcm_fpe_switch|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_denormal_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_cxx_flags|-msse3 -DNDEBUG|g;" "$CONFIG_FILE"
		;;
	    profile)
		sed -i -e "s|lcm_fpe_switch|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_denormal_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_cxx_flags|-msse3 -DNDEBUG|g;" "$CONFIG_FILE"
		;;
	    small)
		sed -i -e "s|lcm_fpe_switch|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_denormal_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_cxx_flags|-msse3 -DNDEBUG|g;" "$CONFIG_FILE"
		;;
            mixed)
		sed -i -e "s|lcm_fpe_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_denormal_switch|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_cxx_flags|-msse3 -std=c++11 -g -O0|g;" "$CONFIG_FILE"
		;;
	    *)
		echo "Unrecognized build type option in config: $BUILD_TYPE"
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
		sed -i -e "s|lcm_enable_hwloc|OFF|g;" "$CONFIG_FILE"
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
		sed -i -e "s|lcm_enable_hwloc|OFF|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_kokkos_devel|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_enable_slfad|ON|g;" "$CONFIG_FILE"
		sed -i -e "s|lcm_slfad_size|-D SLFAD_SIZE=48|g;" "$CONFIG_FILE"
		;;
	    *)
		echo "Unrecognized architecture option in config: $ARCH"
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
	echo "BUILDING $PACKAGE_STRING ..."
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
                    NETCDF_SYSLIB=$NETCDF_LIB/libnetcdf.so
                    NETCDF_LCMLIB="$INSTALL_DIR/lib/libnetcdf.so"
                    ln -sf "$INSTALL_DIR/include" "$INSTALL_DIR/inc"
                    ln -sf "$NETCDF_SYSLIB" "$NETCDF_LCMLIB"
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
		echo "Unrecognized package option in build: $PACKAGE"
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
		echo "Unrecognized package option in test: $PACKAGE"
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
