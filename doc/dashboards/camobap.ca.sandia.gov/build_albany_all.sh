#!/bin/bash

bash nightly_cron_script_albany32bitT.sh; bash process_results_32bitT.sh; bash send_email_32bitT.sh #Albany, 32-bit, No Epetra
bash nightly_cron_script_cismAlbany.sh; bash process_results_cismAlbany.sh; bash send_email_cismAlbany.sh #CISM-Albany, Tpetra
bash nightly_cron_script_albany32bit.sh; bash process_results_32bit.sh; bash send_email_32bit.sh #Albany, 32-bit 
bash nightly_cron_script_albany64bit.sh; bash process_results_64bit.sh; bash send_email_64bit.sh #Albany, 64-bit 
bash nightly_cron_script_albany64bitT.sh; bash process_results_64bitT.sh; bash send_email_64bitT.sh #Albany, 64-bit, No Epetra
bash nightly_cron_script_cismAlbanyEpetra.sh; bash process_results_cismAlbanyEpetra.sh; bash send_email_cismAlbanyEpetra.sh #CISM-Albany, Epetra
bash nightly_cron_script_albanyFunctorOpenMP.sh; bash process_results_functor_openMP.sh; bash send_email_functor_openMP.sh #Albany, OpenMP KokkosNode
~                                                                                                                                                 
~                                              
