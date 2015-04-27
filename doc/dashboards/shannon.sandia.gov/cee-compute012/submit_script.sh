#!/bin/sh

# Crontab entry
#
# Run at 5:00AM local time every day
#
# 00 07 * * * /projects/AppComp/nightly_gahanse/shannon-AlbanyCUVM/submit_script.sh

DOWNLOAD_FILES="True"
#DOWNLOAD_FILES=

UPLOAD_FILES="True"
#UPLOAD_FILES=

BASE_DIR=/projects/AppComp/nightly_gahanse/shannon-AlbanyCUVM
SCRIPT_DIR=/ascldap/users/gahanse/Codes/Albany/doc/dashboards/shannon.sandia.gov/cee-compute012

CDASH_SITE="http://cdash.sandia.gov/CDash-2-3-0/submit.php?project=Albany"

export PATH=/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=/projects/AppComp/nightly_gahanse/shannon-AlbanyCUVM/nightly_log.txt

cd $BASE_DIR

DOWNLOAD_DIR=$BASE_DIR/Download/Albany

echo "Running CDash post at $now" > $LOG_FILE
echo >> $LOG_FILE

if [ "$DOWNLOAD_FILES" ]; then

# Remove the download directory if it exists

 if [ -d "$DOWNLOAD_DIR" ]; then
	echo "rm -rf $DOWNLOAD_DIR" >> $LOG_FILE
    echo >> $LOG_FILE
	rm -rf $DOWNLOAD_DIR
 fi

 echo "mkdir $DOWNLOAD_DIR" >> $LOG_FILE
 echo >> $LOG_FILE
 mkdir $DOWNLOAD_DIR

# rsync the files
  echo "rsync -a gahanse@software-login.sandia.gov:/home/gahanse/Albany/ $DOWNLOAD_DIR" >> $LOG_FILE
  echo >> $LOG_FILE
  rsync -a --delete gahanse@software-login.sandia.gov:/home/gahanse/Albany/ $DOWNLOAD_DIR >> $LOG_FILE 2>&1

fi

if [ "$UPLOAD_FILES" ]; then

# curl the Project.xml file to the CDash site
  echo "Sending Project.xml to CDash site: $CDASH_SITE" >> $LOG_FILE
  echo >> $LOG_FILE
  curl -T $SCRIPT_DIR/Project.xml $CDASH_SITE >> $LOG_FILE 2>&1
  echo >> $LOG_FILE


# curl the files to the CDash site

  if [ -d "$DOWNLOAD_DIR" ]; then

   for files in $DOWNLOAD_DIR/*; do
      echo "Sending $files to CDash site: $CDASH_SITE" >> $LOG_FILE
      curl -T $files $CDASH_SITE >> $LOG_FILE 2>&1
      echo >> $LOG_FILE
   done
  fi

fi

echo >> $LOG_FILE
echo "Done!!!" >> $LOG_FILE
