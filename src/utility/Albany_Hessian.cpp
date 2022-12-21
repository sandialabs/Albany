
#include "Albany_Hessian.hpp"
#include "Albany_KokkosUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_StringUtils.hpp"

namespace Albany
{

Teuchos::RCP<MatrixBased_LOWS>
createDenseHessianLinearOp(Teuchos::RCP<const Thyra_VectorSpace> p_vs)
{
  Teuchos::RCP<const Tpetra_Map> p_map = getTpetraMap(p_vs);
  Teuchos::RCP<Thyra_LinearOp> H;

  Tpetra_GO num_params = p_map->getLocalNumElements();

  Teuchos::RCP<Tpetra_CrsGraph> Hgraph = Teuchos::rcp(new Tpetra_CrsGraph(p_map, num_params));

  Teuchos::Array<Tpetra_GO> cols(num_params);

  for (Tpetra_GO iparam = 0; iparam < num_params; ++iparam)
  {
    cols[iparam] = p_map->getGlobalElement(iparam);
  }

  for (Tpetra_GO iparam = 0; iparam < num_params; ++iparam)
  {
    Hgraph->insertGlobalIndices(cols[iparam], num_params, cols.getRawPtr());
  }

  Hgraph->fillComplete();
  Teuchos::RCP<Tpetra_CrsMatrix> Ht = Teuchos::rcp(new Tpetra_CrsMatrix(Hgraph));

  H = createThyraLinearOp(Ht);
  assign(H, 0.0);

  return Teuchos::rcp(new MatrixBased_LOWS(H));
}

Teuchos::RCP<Thyra_LinearOp>
createSparseHessianLinearOp(const Teuchos::RCP<const DistributedParameter>& p)
{
  const auto p_owned_vs = p->vector()->space();
  const auto p_overlapped_vs = p->overlapped_vector()->space();

  const auto p_owned_map = getTpetraMap(p_owned_vs);
  const auto p_overlapped_map = getTpetraMap(p_overlapped_vs);

  Teuchos::RCP<Thyra_LinearOp> H;

  const auto p_dof_mgr = p->get_dof_mgr();
  const auto p_elem_dof_lids = p_dof_mgr->elem_dof_lids().host();
  const int nelem = p_elem_dof_lids.extent(0);
  const int ndofs = p_elem_dof_lids.extent(1);

  auto Hgraph = Teuchos::rcp(new Tpetra_CrsGraph(p_owned_map, 30));

  Tpetra_GO cols[1];
  for (int ielem=0; ielem<nelem; ++ielem) {
    for (int idof=0; idof<ndofs; ++idof) {
      const Tpetra_LO ov_node1 = p_elem_dof_lids(ielem,idof);

      const Tpetra_GO row = p_overlapped_map->getGlobalElement(ov_node1);
      for (int jdof=0; jdof<ndofs; ++jdof) {
        const Tpetra_LO ov_node2 = p_elem_dof_lids(ielem,jdof);

        cols[0] = p_overlapped_map->getGlobalElement(ov_node2);
        Hgraph->insertGlobalIndices(row,1,cols);
      }
    }
  }

  Hgraph->fillComplete();
  auto Ht = Teuchos::rcp(new Tpetra_CrsMatrix(Hgraph));

  H = createThyraLinearOp(Ht);
  assign(H, 0.0);
  return Teuchos::rcp(new MatrixBased_LOWS(H));
}

void getHessianBlockIDs(int &i1, int &i2, std::string blockName)
{
  std::string tmp = blockName;
  tmp.erase(std::remove(tmp.begin(), tmp.end(), '('), tmp.end());
  tmp.erase(std::remove(tmp.begin(), tmp.end(), ')'), tmp.end());

  std::vector<std::string> block_ids;

  util::splitStringOnDelim(tmp, ',', block_ids);
  int ids[2];

  for (int i = 0; i < 2; ++i)
  {
    if (block_ids[i][0] == 'x')
      ids[i] = 0;
    else if (block_ids[i][0] == 'p')
      ids[i] = stoi(block_ids[i].substr(1)) + 1;
    else
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameter,
          std::endl
              << "Error!  Albany::getHessianBlockIDs():  "
              << "The name " << blockName
              << " is incorrect; it is impossible to deduce if "
              << block_ids[i]
              << " refers to a parameter or the solution."
              << std::endl);
  }

  i1 = ids[0];
  i2 = ids[1];
}

void getParameterVectorID(int &i,
                          bool &is_distributed,
                          std::string parameterName)
{
  std::vector<std::string> elems;
  util::splitStringOnDelim(parameterName, ' ', elems);
  if (elems.size() == 2 && elems[0].compare("parameter_vector") == 0)
  {
    is_distributed = false;
    i = stoi(elems[1]);
  } else {
    is_distributed = true;
    i = -1;
  }
}

} // namespace Albany
