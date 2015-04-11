#!/bin/sh

# Crontab entry
#
# Run at 5:00AM local time every day
#
# 00 05 * * * /ascldap/users/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/cee-compute012/script_driver.sh

export PATH=/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin

# Note: shannon does not let one ssh with a public key, so we have to do a keytab to obfiscate a kerberos key

# Generate the keytab:

# > ktutil
# ktutil:  addent -password -p gahanse@dce.sandia.gov -k 1 -e rc4-hmac
# Password for gahanse@dce.sandia.gov:
# ktutil:  wkt gahanse.keytab
# ktutil:  quit

# Do the kinit

#echo "kinit gahanse@dce.sandia.gov -k -t /ascldap/users/gahanse/.ssh/gahanse.keytab" >> $LOG_FILE
#echo >> $LOG_FILE

kinit gahanse@dce.sandia.gov -k -t /ascldap/users/gahanse/.ssh/gahanse.keytab

# Call all the scripts to download from shannon
#
#
# Minicontact

/ascldap/users/gahanse/Codes/ACME/ACME_miniContact/doc/dashboards/shannon.sandia.gov/cee-compute011/submit_script.sh

# Albany

/ascldap/users/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/cee-compute012/submit_script.sh

# kill the ticket
#  echo "kdestroy" >> $LOG_FILE
#  echo >> $LOG_FILE

kdestroy

#echo >> $LOG_FILE
#echo "Done!!!" >> $LOG_FILE
