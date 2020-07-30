
#!/bin/bash
awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-sems-intel >& /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i "s/\"/'/g" /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i 's/\.\.//g' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i 's,\\,,g' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i '/^$/d' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i 's/-D /"-D/g' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
awk '{print $0 "\""}' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt >& tmp.txt
mv tmp.txt /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i 's, \",\",g' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i '$ d' /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
cat /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt
