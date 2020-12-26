#!/bin/bash

rm -rf repos
rm -rf build
rm -rf nightly_log*
rm -rf results*
bash nightly_cron_script_albany.sh; bash process_results.sh #Albany
bash nightly_cron_script_cismAlbany.sh; bash process_results_cismAlbany.sh #CISM-Albany
bash nightly_cron_script_albanyT.sh; bash process_resultsT.sh #Albany, No Epetra
bash nightly_cron_script_albanyKokkosNodeOpenMP.sh; bash process_results_functor_openMP.sh #Albany, OpenMP KokkosNode
#IKT, 12/24/2020 - the following is a hack since configure does not work correctly using cmake/cdash scripts
mkdir build/IKTAlbanyFPECheckDbg
cp do-cmake-fpe-check build/IKTAlbanyFPECheckDbg
cd build/IKTAlbanyFPECheckDbg
./do-cmake-fpe-check
cd /nightlyCDash
bash nightly_cron_script_albany_fpe.sh; bash process_results_fpe.sh #Albany FPE check on 
#IKT, 12/26/2020 - the following is a hack that does configure after the code is built
#to avoid missing configure in nightlies
bash nightly_cron_script_albany_fpe_config.sh
~                                                                                                                                                 
~                                              
