#!/bin/bash

bash nightly_cron_script_albanyT.sh; bash process_resultsT.sh; bash send_emailT.sh #Albany, No Epetra
bash nightly_cron_script_cismAlbany.sh; bash process_results_cismAlbany.sh; bash send_email_cismAlbany.sh #CISM-Albany, Tpetra
bash nightly_cron_script_albany.sh; bash process_results.sh; bash send_email.sh #Albany
bash nightly_cron_script_albany_fpe.sh; bash process_results_fpe.sh; bash send_email_fpe.sh #Albany FPE check on 
bash nightly_cron_script_cismAlbanyEpetra.sh; bash process_results_cismAlbanyEpetra.sh; bash send_email_cismAlbanyEpetra.sh #CISM-Albany, Epetra
bash nightly_cron_script_albanyFunctorOpenMP.sh; bash process_results_functor_openMP.sh; bash send_email_functor_openMP.sh #Albany, OpenMP KokkosNode
~                                                                                                                                                 
~                                              
