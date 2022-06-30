#include "Albany_TpetraTypes.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_Hessian.hpp"
#include "Albany_KokkosUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_StringUtils.hpp"

#include <Teuchos_RCP.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Sort.hpp>

#include <math.h>

namespace Albany
{

Teuchos::RCP<Thyra_LinearOp>
createDenseHessianLinearOp(Teuchos::RCP<const Thyra_VectorSpace> p_vs)
{
    Teuchos::RCP<const Tpetra_Map> p_map = getTpetraMap(p_vs);
    Teuchos::RCP<Thyra_LinearOp> H;

    Tpetra_GO num_params = p_map->getLocalNumElements();

    Teuchos::RCP<Tpetra_CrsGraph> Hgraph = Teuchos::rcp(new Tpetra_CrsGraph(p_map, num_params));

    Teuchos::Array<Tpetra_GO> cols(num_params);

    for (Tpetra_GO iparam=0; iparam<num_params; ++iparam) {
        cols[iparam] = p_map->getGlobalElement(iparam);
    }

    for (Tpetra_GO iparam=0; iparam<num_params; ++iparam) {
        Hgraph->insertGlobalIndices(cols[iparam], num_params, cols.getRawPtr());
    }

    Hgraph->fillComplete();
    Teuchos::RCP<Tpetra_CrsMatrix> Ht = Teuchos::rcp(new Tpetra_CrsMatrix(Hgraph));

    H = createThyraLinearOp(Ht);
    assign(H, 0.0);

    return H;
}

// Teuchos::RCP<Thyra_LinearOp>
// createSparseHessianLinearOp(
//     Teuchos::RCP<const Thyra_VectorSpace> p_owned_vs,
//     Teuchos::RCP<const Thyra_VectorSpace> p_overlapped_vs,
//     const std::vector<IDArray> vElDofs)
// {
//     Teuchos::RCP<const Tpetra_Map> p_overlapped_map = getTpetraMap(p_overlapped_vs);
//     Teuchos::RCP<const Tpetra_Map> p_owned_map = getTpetraMap(p_owned_vs);
//     Teuchos::RCP<Thyra_LinearOp> H;

//     std::size_t num_elem = 0;
//     const std::size_t num_elem_per_ws = vElDofs[0].dimension(0);
//     const std::size_t nws = vElDofs.size();

//     bool same_num_elem_per_ws = true;

//     for (std::size_t wsIndex = 0; wsIndex < nws; ++wsIndex)
//     {
//         const std::size_t num_elem_per_ws_i = vElDofs[wsIndex].dimension(0);
//         if (num_elem_per_ws != num_elem_per_ws_i && wsIndex + 1 < nws)
//             same_num_elem_per_ws = false;
//         num_elem += vElDofs[wsIndex].dimension(0);
//     }

//     TEUCHOS_TEST_FOR_EXCEPTION(
//         same_num_elem_per_ws == false,
//         std::logic_error,
//         std::endl
//             << "Error!  Albany::createHessianCrsGraph():  "
//             << "Not implemented yet"
//             << std::endl);

//     const std::size_t NN = vElDofs[0].dimension(1);

//     Teuchos::RCP<Tpetra_CrsGraph> Hgraph = Teuchos::rcp(new Tpetra_CrsGraph(p_owned_map, 30));

//     Tpetra_GO cols[1];

//     for (std::size_t ielem=0; ielem<num_elem; ++ielem) {
//         IDArray wsElDofs = vElDofs[floor(ielem / num_elem_per_ws)];
//         const Tpetra_LO ielem_ws = ielem % num_elem_per_ws;
//         for (std::size_t i = 0; i < NN; ++i)
//         {
//             const Tpetra_LO lcl_overlapped_node1 = wsElDofs((int)ielem_ws, (int)i, 0);
//             if (lcl_overlapped_node1 < 0)
//                 continue;

//             const GO row = p_overlapped_map->getGlobalElement(lcl_overlapped_node1);

//             for (std::size_t j = 0; j < NN; ++j)
//             {
//                 const Tpetra_LO lcl_overlapped_node2 = wsElDofs((int)ielem_ws, (int)j, 0);
//                 if (lcl_overlapped_node2 < 0)
//                     continue;

//                 cols[0] = p_overlapped_map->getGlobalElement(lcl_overlapped_node2);

//                 Hgraph->insertGlobalIndices(row, 1, cols);
//             }
//         }
//     }

//     Hgraph->fillComplete();
//     Teuchos::RCP<Tpetra_CrsMatrix> Ht = Teuchos::rcp(new Tpetra_CrsMatrix(Hgraph));

//     H = createThyraLinearOp(Ht);
//     assign(H, 0.0);

//     return H;
// }

Teuchos::RCP<Thyra_LinearOp>
createSparseHessianLinearOp(
    const Teuchos::RCP<const DistributedParameter>& param,
    const Teuchos::RCP<const AbstractDiscretization>& disc)

    // return createSparseHessianLinearOp(distParamLib->get(param_name),
    //                                    app->getDisc());
    // Teuchos::RCP<const Thyra_VectorSpace> p_owned_vs,
    // Teuchos::RCP<const Thyra_VectorSpace> p_overlapped_vs,
    // Teuchos::RCP<const panzer::DOFManager> p_dof_mgr,
    // const WorksetArray<int>& ws_sizes,
    // const DualView<int*>& ws_elem_lids)
{
    auto p_space    = param->vector_space();
    auto p_ov_space = param->overlap_vector_space();
    // Teuchos::RCP<const Tpetra_Map> p_overlapped_map = getTpetraMap(p_overlapped_vs);
    // Teuchos::RCP<const Tpetra_Map> p_owned_map = getTpetraMap(p_owned_vs);

    // Teuchos::RCP<Thyra_LinearOp> H;

    // int num_elem = 0;
    // const int num_elem_per_ws = ws_sizes[0];
    // const int nws = ws_sizes.size();

    // bool same_num_elem_per_ws = true;

    // for (int iws = 0; iws < nws; ++iws) {
    //     const int num_elem_per_ws_i = ws_sizes[iws];
    //     if (num_elem_per_ws != num_elem_per_ws_i && iws + 1 < nws)
    //         same_num_elem_per_ws = false;
    //     num_elem += ws_sizes[iws];
    // }

    // TEUCHOS_TEST_FOR_EXCEPTION(
    //     same_num_elem_per_ws == false,
    //     std::logic_error,
    //     std::endl
    //         << "Error!  Albany::createHessianCrsGraph():  "
    //         << "Not implemented yet"
    //         << std::endl);

    ThyraCrsMatrixFactory H_factory (p_space,p_space,p_ov_space,p_ov_space);
    
    // Teuchos::RCP<Tpetra_CrsGraph> Hgraph = Teuchos::rcp(new Tpetra_CrsGraph(p_owned_map, 30));
    Teuchos::Array<GO> cols(1);

    // Loop over all elems in the param dof mgr
    const auto& p_dof_mgr = param->dof_mgr();
    const auto& conn_mgr  = p_dof_mgr()->getConnManager();

    // Local ids of elements
    const auto& elem_lids = conn_mgr->getElementBlock(param->ebName());
    // const int   num_elem  = elem_lids.size();

    const auto p_ov_gids = getGlobalElements(p_ov_space);

    // Global ids of dofs in the element
    for (const auto elem_lid : elem_lids) {
        // int ws       = ielem / num_elem_per_ws;
        // int ielem_ws = ielem % num_elem_per_ws; 
        // int elem_lid = ws_elem_lids(ws,ielem_ws);
        // IDArray wsElDofs = vElDofs[floor(ielem / num_elem_per_ws)];
        // const Tpetra_LO ielem_ws = ielem % num_elem_per_ws;
      std::vector<Tpetra_GO> elem_gids;
      p_dof_mgr->getElementGIDs(elem_lid,elem_gids);
      std::vector<Tpetra_GO> param_elem_gids;
      for (const auto gid : elem_gids) {
        if (std::find(p_ov_gids.begin(),p_ov_gids.end(),gid)!=p_ov_gids.end()) {
          param_elem_gids.push_back(gid);
        }
      }
      for (const auto row : param_elem_gids) {
        for (const auto col : param_elem_gids) {
          cols[0] = col;
          H_factory.insertGlobalIndices(row,cols());
        }
      }
    }

    H_factory.fillComplete();
    auto H = H_factory.createOp ();
    assign(H, 0.0);

    return H;
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

void getParameterVectorID(
    int &i,
    bool &is_distributed,
    std::string parameterName)
{
    std::vector<std::string> elems;
    util::splitStringOnDelim(parameterName, ' ', elems);
    if (elems.size() == 2 && elems[0].compare("parameter_vector") == 0)
    {
        is_distributed = false;
        i = stoi(elems[1]);
    }
    else
    {
        is_distributed = true;
        i = -1;
    }
}

} // namespace Albany
