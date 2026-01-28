#!/usr/bin/env bash
set -e

# useful env changes
export CLICOLOR=1
# make prompt show time, user@where, and current path
export PS1="\[\e[32;1m\]\t \u@\h \[\e[0m\]\[\e[33;1m\]\w\[\e[0m\]\[\e[32;1m\]\$\[\e[0m\] "
bind "\"\e[A\": history-search-backward"
bind "\"\e[B\": history-search-forward"
git config --global url."https://github.com/".insteadOf git@github.com:

# Load compass
source load_compass_1.8.0_mpich.sh
echo "compass conda env activated"

# Execute the command passed to the container
exec "$@"
