This page documents the HEAD commits that were used in the last successful nightly
dashdoard test posted at:

https://my.cdash.org/index.php?project=Albany

for the Albany64BitClang dashboard entry shown there. Note that these commits rely on
100% of the tests passing, a single test failure will prevent these SHAs from being 
updated.

Trilinos (develop branch):

@TRILINOS_HEAD_SHA@

Albany:

@ALBANY_HEAD_SHA@

To sync to these commits use the following command either in the Albany or Trilinos directories:

git reset --hard [commit SHA]
