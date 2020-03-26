#!/bin/bash

bash nightly_cron_script_albany.sh; bash process_results.sh #Albany
bash nightly_cron_script_albanyT.sh; bash process_resultsT.sh #Albany, No Epetra
bash nightly_cron_script_albanyKokkosNodeOpenMP.sh; bash process_results_functor_openMP.sh #Albany, OpenMP KokkosNode
bash nightly_cron_script_albany_fpe.sh; bash process_results_fpe.sh #Albany FPE check on 
~                                                                                                                                                 
~                                              
