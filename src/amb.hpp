// Some simple debug code so we can compare vanilla Albany with Kokkosified
// Albany.

#ifndef INCLUDE_AMB
#define INCLUDE_AMB

#include <stdio.h>
#include <iostream>
#include <sstream>
#include "Albany_DataTypes.hpp"

namespace amb {
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Array;
using namespace std;

string get_filename_with_int(const string& filename, int i);
string get_mpi_filename(const string& id);
string kokkos_dec(const string& str);
string get_full_filename(const string& str);

void write_multivector(
  const RCP<const Tpetra_MultiVector>& v, const string& filename);

void write_matrix(
  const RCP<const Tpetra_CrsMatrix>& a, const string& filename);

void set_global_int(int i, int v);
int get_global_int(int i);
void incr_global_int(int i);
enum { gi_res = 0, gi_ws = 1, gi_jac = 2, gi_magic = 4 };

int print_level();
bool set_xT_for_debug();
bool set_own_omp_nthreads();

// MDField ugliness. I can't think of how to do an index iterator for MDField
// that I can then use in operator(), so I'll do macros for now.
#define amb_write_mdfield2(f, filename, n1, n2)                         \
  do {                                                                  \
    FILE* fid = fopen(amb::get_full_filename(filename).c_str(), "wa");  \
    for (int i1 = 0; i1 < n1; ++i1)                                     \
      for (int i2 = 0; i2 < n2; ++i2)                                   \
        fprintf(fid, "%d %d %1.15e\n", i1, i2, f(i1, i2));              \
    fclose(fid);                                                        \
  } while (0);
#define amb_write_mdfield3(f, filename, n1, n2, n3)                     \
  do {                                                                  \
    FILE* fid = fopen(amb::get_full_filename(filename).c_str(), "wa");  \
    for (int i1 = 0; i1 < n1; ++i1)                                     \
      for (int i2 = 0; i2 < n2; ++i2)                                   \
        for (int i3 = 0; i3 < n3; ++i3)                                 \
          fprintf(fid, "%d %d %d %1.15e\n", i1, i2, i3, f(i1, i2, i3)); \
    fclose(fid);                                                        \
  } while (0);
#define amb_write_mdfield4(f, filename, n1, n2, n3, n4)                 \
  do {                                                                  \
    FILE* fid = fopen(amb::get_full_filename(filename).c_str(), "wa");  \
    for (int i1 = 0; i1 < n1; ++i1)                                     \
      for (int i2 = 0; i2 < n2; ++i2)                                   \
        for (int i3 = 0; i3 < n3; ++i3)                                 \
          for (int i4 = 0; i4 < n4; ++i4)                               \
            fprintf(fid, "%d %d %d %d %1.15e\n", i1, i2, i3, i4,        \
                    f(i1, i2, i3, i4));                                 \
    fclose(fid);                                                        \
  } while (0);
} // namespace amb

#endif // INCLUDE_AMB
