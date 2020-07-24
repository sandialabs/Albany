#!/bin/bash

sed -i -e '273s,^,/* ,' /projects/albany/nightlyAlbanyCDash/repos/Trilinos/packages/tpetra/core/src/Tpetra_FECrsGraph_def.hpp
sed -i -e '293s,^,*/ ,' /projects/albany/nightlyAlbanyCDash/repos/Trilinos/packages/tpetra/core/src/Tpetra_FECrsGraph_def.hpp
