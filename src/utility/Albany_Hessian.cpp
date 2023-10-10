#include "Albany_Hessian.hpp"

#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_StringUtils.hpp"

namespace Albany
{

Teuchos::RCP<MatrixBased_LOWS>
createDenseHessianLinearOp(Teuchos::RCP<const Thyra_VectorSpace> p_vs)
{
  ThyraCrsMatrixFactory H_factory(p_vs,p_vs);
  auto gids = getGlobalElements(p_vs);
  H_factory.insertGlobalIndices(gids,gids,false);
  H_factory.fillComplete();

  auto H = H_factory.createOp();
  assign(H, 0.0);

  return Teuchos::rcp(new MatrixBased_LOWS(H));
}

Teuchos::RCP<Thyra_LinearOp>
createSparseHessianLinearOp(const Teuchos::RCP<const DistributedParameter>& p)
{
  const auto p_owned_vs = p->vector()->space();
  const auto p_overlapped_vs = p->overlapped_vector()->space();

  const auto p_dof_mgr = p->get_dof_mgr();
  const int nelem = p_dof_mgr->getAlbanyConnManager()->getElementBlock().size();

  ThyraCrsMatrixFactory H_factory(p_owned_vs,p_owned_vs,p_overlapped_vs,p_overlapped_vs);

  for (int ielem=0; ielem<nelem; ++ielem) {
    const auto& p_elem_gids = p_dof_mgr->getElementGIDs(ielem);
    std::vector<GO> valid_gids;
    for (auto gid : p_elem_gids) {
      if (gid>=0)
        valid_gids.push_back(gid);
    }
    // Don't say symmetric=true, since you're already doing it by passing rows=cols.
    H_factory.insertGlobalIndices(valid_gids,valid_gids,false);
  }
  H_factory.fillComplete();

  auto H = H_factory.createOp();
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
