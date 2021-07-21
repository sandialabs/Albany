
#include <Teuchos_RCP.hpp>
#include "Albany_TpetraTypes.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_Hessian.hpp"
#include "Albany_KokkosUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"

#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Sort.hpp>
#include <math.h>

Teuchos::RCP<Thyra_LinearOp> Albany::createHessianLinearOp(
    Teuchos::RCP<const Thyra_VectorSpace> p_owned_vs,
    Teuchos::RCP<const Thyra_VectorSpace> p_overlapped_vs,
    const std::vector<IDArray> vElDofs)
{
    Teuchos::RCP<const Tpetra_Map> p_overlapped_map = Albany::getTpetraMap(p_overlapped_vs);
    Teuchos::RCP<const Tpetra_Map> p_owned_map = Albany::getTpetraMap(p_owned_vs);
    Teuchos::RCP<Thyra_LinearOp> H;

    std::size_t num_elem = 0;
    const std::size_t num_elem_per_ws = vElDofs[0].dimension(0);
    const std::size_t nws = vElDofs.size();

    bool same_num_elem_per_ws = true;

    for (std::size_t wsIndex = 0; wsIndex < nws; ++wsIndex)
    {
        const std::size_t num_elem_per_ws_i = vElDofs[wsIndex].dimension(0);
        if (num_elem_per_ws != num_elem_per_ws_i && wsIndex + 1 < nws)
            same_num_elem_per_ws = false;
        num_elem += vElDofs[wsIndex].dimension(0);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
        same_num_elem_per_ws == false,
        std::logic_error,
        std::endl
            << "Error!  Albany::createHessianCrsGraph():  "
            << "Not implemented yet"
            << std::endl);

    const std::size_t NN = vElDofs[0].dimension(1);

    Teuchos::RCP<Tpetra_CrsGraph> Hgraph = Teuchos::rcp(new Tpetra_CrsGraph(p_owned_map, 30));

    Tpetra_GO cols[1];
    Teuchos::Array<ST> vals(1);

    for (std::size_t ielem; ielem<num_elem; ++ielem) {
        IDArray wsElDofs = vElDofs[floor(ielem / num_elem_per_ws)];
        const Tpetra_LO ielem_ws = ielem % num_elem_per_ws;
        for (std::size_t i = 0; i < NN; ++i)
        {
            const Tpetra_LO lcl_overlapped_node1 = wsElDofs((int)ielem_ws, (int)i, 0);
            if (lcl_overlapped_node1 < 0)
                continue;

            const GO row = p_overlapped_map->getGlobalElement(lcl_overlapped_node1);

            for (std::size_t j = 0; j < NN; ++j)
            {
                const Tpetra_LO lcl_overlapped_node2 = wsElDofs((int)ielem_ws, (int)j, 0);
                if (lcl_overlapped_node2 < 0)
                    continue;

                cols[0] = p_overlapped_map->getGlobalElement(lcl_overlapped_node2);

                Hgraph->insertGlobalIndices(row, 1, cols);
            }
        }
    }

    Hgraph->fillComplete();
    Teuchos::RCP<Tpetra_CrsMatrix> Ht = Teuchos::rcp(new Tpetra_CrsMatrix(Hgraph));

    H = Albany::createThyraLinearOp(Ht);
    assign(H, 0.0);

    return H;
}

void Albany::getHessianBlockIDs(
    int &i1,
    int &i2,
    std::string blockName)
{
    std::string tmp = blockName;
    tmp.erase(std::remove(tmp.begin(), tmp.end(), '('), tmp.end());
    tmp.erase(std::remove(tmp.begin(), tmp.end(), ')'), tmp.end());

    std::vector<std::string> block_ids;

    Albany::splitStringOnDelim(tmp, ',', block_ids);
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

void Albany::getParameterVectorID(
    int &i,
    bool &is_distributed,
    std::string parameterName)
{
    std::vector<std::string> elems;
    Albany::splitStringOnDelim(parameterName, ' ', elems);
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
