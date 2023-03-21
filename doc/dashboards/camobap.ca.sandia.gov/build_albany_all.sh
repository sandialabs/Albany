#!/bin/bash

export PATH=/usr/bin:$PATH:/tpls/install/ninja/build-cmake
alias cmake=/usr/bin/cmake
echo cmake --version
rm -rf repos
rm -rf build
rm -rf nightly_log*
rm -rf results*
bash nightly_cron_script_albany.sh #Albany
bash nightly_cron_script_cismAlbany.sh #CISM-Albany
#bash nightly_cron_script_albanyT.sh #Albany, No Epetra
bash nightly_cron_script_albany_fpe.sh #Albany FPE check on
bash nightly_cron_script_albany_openmp.sh #Albany, OpenMP KokkosNode
#bash nightly_cron_script_trilinos.sh #Trilinos with extended scalar types                                         
#bash nightly_cron_script_trilinos_eti_ld.sh #Trilinos with long double and ETI 
#bash nightly_cron_script_alegra-xfem.sh; bash process_results_alegra-xfem.sh; bash send_email_alegra-xfem.sh
#bash process_results_alegra-xfem_eti.sh; bash send_email_alegra-xfem_eti.sh
~                                              
