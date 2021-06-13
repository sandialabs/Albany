#!/bin/bash

rm -rf repos
rm -rf build
rm -rf nightly_log*
rm -rf results*
bash nightly_cron_script_albany.sh; bash process_results.sh #Albany
bash nightly_cron_script_cismAlbany.sh; bash process_results_cismAlbany.sh #CISM-Albany
bash nightly_cron_script_albanyT.sh; bash process_resultsT.sh #Albany, No Epetra
bash nightly_cron_script_albanyKokkosNodeOpenMP.sh; bash process_results_functor_openMP.sh #Albany, OpenMP KokkosNode
bash nightly_cron_script_albany_fpe.sh; bash process_results_fpe.sh #Albany FPE check on
bash nightly_cron_script_trilinos.sh #Trilinos with extended scalar types                                                    
bash nightly_cron_script_alegra-xfem.sh; bash process_results_alegra-xfem.sh; bash send_email_alegra-xfem.sh
~                                              
