cd /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier

# This file is sourced from cron, so use if/else rather than exit.
# If last night's test jobs are still queued or running, skip tonight: cleaning
# and rebuilding would delete/overwrite the builds those jobs are using.
PENDING_JOBS=$(squeue -u ${USER:-$(whoami)} -h -n AlbanyTesting,AlbanyPerfTesting -o "%i %j %T" 2>/dev/null)
if [ -n "$PENDING_JOBS" ]; then
  echo "$(date): skipping nightly build/test, previous jobs still in queue:" >> nightly_skipped.log
  echo "$PENDING_JOBS" >> nightly_skipped.log
else
  source /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier/clean-up.sh
  source /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier/run_all_jobs.sh
fi
