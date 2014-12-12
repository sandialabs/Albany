#include "amb.hpp"
#include <vector>

namespace amb {
//todo Make this work in general. Might be hard/expensive.
inline Tpetra::global_size_t
getGlobalNumUniqueElements (const RCP<const Tpetra_Map>& map) {
  return map->getMaxAllGlobalIndex() - map->getMinAllGlobalIndex() + 1;
}

string get_mpi_filename (const string& id) {
  RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
  stringstream ss;
  ss << id << "_" << comm->getRank() << ".dat";
  return ss.str();
}

string get_filename_with_int (const string& filename, int i) {
  stringstream ss;
  ss << filename << "_" << i;
  return ss.str();
}

string get_full_filename (const string& base) {
  return kokkos_dec(
    get_mpi_filename(
      get_filename_with_int(
        get_filename_with_int(base, get_global_int(gi_res)),
        get_global_int(gi_ws))));
}

string kokkos_dec (const string& str) {
  return
#ifdef AMB_KOKKOS
    "k_" +
#else
    "v_" +
#endif
    str;
}

void write_multivector (
  const RCP<const Tpetra_MultiVector>& v, const string& filename)
{
  if (v->getMap()->getComm()->getRank() == 0)
    std::cout << "amb: writing " << filename << std::endl;
  FILE* fid = fopen(kokkos_dec(get_mpi_filename(filename)).c_str(), "wa");
  RCP<const Tpetra_Map> map = v->getMap();
  for (size_t ir = 0; ir < v->getLocalLength(); ++ir) {
    GO gid = map->getGlobalElement(ir);
    fprintf(fid, " %10d", gid);
    for (size_t ic = 0; ic < v->getNumVectors(); ++ic)
      fprintf(fid, " %1.18e", v->getData(ic)[ir]);
    fprintf(fid, "\n");
  }
  fclose(fid);
}

template <class SparseMatrixType>
void write_mm_header (FILE* fid, const RCP<SparseMatrixType>& a)
{
  fprintf(fid, "%%%%MatrixMarket matrix coordinate real general\n"
          "%11ld %11ld %11ld\n",
          getGlobalNumUniqueElements(a->getRangeMap()),
          getGlobalNumUniqueElements(a->getDomainMap()),
          a->getNodeNumEntries());
}

void write_matrix (
  const RCP<const Tpetra_CrsMatrix>& a, const string& filename)
{
  if (a->getRowMap()->getComm()->getRank() == 0)
    std::cout << "amb: writing " << filename << std::endl;
  FILE* fid = fopen(kokkos_dec(get_mpi_filename(filename)).c_str(), "wa");
  write_mm_header(fid, a);
  RCP<const Tpetra_Map> map = a->getRowMap();
  Array<GO> col(a->getGlobalMaxNumRowEntries());
  Array<ST> val(a->getGlobalMaxNumRowEntries());
  size_t ne;
  for (int i = 0; i < a->getNodeNumRows(); ++i) {
    GO gid = map->getGlobalElement(i);
    a->getGlobalRowCopy(gid, col, val, ne);
    for (int j = 0; j < ne; ++j)
      fprintf(fid, "%11ld %11ld %25.18e\n", gid+1, col[j]+1,
              val[j] == 0 ? 42e-42 : val[j]);
  }
  fclose(fid);
}

static int& _get_global_int (int i) {
  static vector<int> gis;
  if (i >= gis.size()) gis.resize(i+1);
  return gis[i];
}
void set_global_int (int i, int v) { _get_global_int(i) = v; }
int get_global_int (int i) { return _get_global_int(i); }
void incr_global_int (int i) { ++_get_global_int(i); }

int print_level () { return 0; }
bool set_xT_for_debug () { return false; }
bool set_own_omp_nthreads () { return false; }
} // namespace amb
