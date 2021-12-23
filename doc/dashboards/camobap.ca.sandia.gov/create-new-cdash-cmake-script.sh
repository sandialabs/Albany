#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /mnt/encrypted_sdc1/nightlyCDash/ctest_nightly_trilinos_tmp.cmake >& /mnt/encrypted_sdc1/nightlyCDash/ctest_nightly_trilinos.cmake
