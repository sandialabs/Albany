
#include <Teuchos_RCP.hpp>
#include "Albany_TpetraTypes.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_Hessian.hpp"

#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Sort.hpp>

int min(int a, int b)
{
    return (a <= b) ? a : b;
}

int max(int a, int b)
{
    return (a >= b) ? a : b;
}

Teuchos::RCP<Tpetra_CrsGraph> Albany::createHessianCrsGraph(
    Teuchos::RCP<const Tpetra_Map> p_owned_map,
    Teuchos::RCP<const Tpetra_Map> p_overlapped_map,
    const Albany::IDArray &wsElDofs)
{
    /*
        Implemented using:

        Hoemmen, M. F., & Edwards, H. C. (2014). Threaded construction and
        fill of Tpetra sparse linear system using Kokkos (No. SAND2014-19125C).
        Sandia National Lab.(SNL-NM), Albuquerque, NM (United States).        
    */
    using std::pair;
    using Teuchos::RCP;
    using Teuchos::rcp;

    const std::size_t num_elem = wsElDofs.dimension(0);
    const std::size_t NN = wsElDofs.dimension(1);

    const int num_rows = p_owned_map->getNodeNumElements();

    Kokkos::View<size_t *> rowCounts("row counts", num_rows);

    const int max_connections = 30;
    Kokkos::UnorderedMap<pair<Tpetra_GO, Tpetra_GO>, bool> nodenode(num_rows * max_connections);

    // Generate elements’ unique node-node pairs
    Kokkos::parallel_for(
        num_elem,
        KOKKOS_LAMBDA(const Tpetra_LO ielem) {
            for (std::size_t i = 0; i < NN; ++i)
            {
                const Tpetra_LO lcl_overlapped_node1 = wsElDofs((int)ielem, (int)i, 0);
                if (lcl_overlapped_node1 < 0)
                    continue;

                const Tpetra_GO global_overlapped_node1 = p_overlapped_map->getGlobalElement(lcl_overlapped_node1);

                for (std::size_t j = i; j < NN; ++j)
                {
                    const Tpetra_LO lcl_overlapped_node2 = wsElDofs((int)ielem, (int)j, 0);
                    if (lcl_overlapped_node2 < 0)
                        continue;

                    const Tpetra_GO global_overlapped_node2 = p_overlapped_map->getGlobalElement(lcl_overlapped_node2);

                    const Tpetra_GO globalNode1 = min(global_overlapped_node1, global_overlapped_node2);
                    const Tpetra_GO globalNode2 = max(global_overlapped_node1, global_overlapped_node2);

                    const pair<Tpetra_GO, Tpetra_GO> key(globalNode1, globalNode2);
                    auto result = nodenode.insert(key);

                    if (result.success())
                    {
                        const bool isDiffNodes = (globalNode1 != globalNode2);
                        const bool isLocalNode1 = p_owned_map->isNodeGlobalElement(globalNode1);
                        const bool isLocalNode2 = p_owned_map->isNodeGlobalElement(globalNode2);

                        const Tpetra_LO lclNode1 = p_owned_map->getLocalElement(globalNode1);
                        const Tpetra_LO lclNode2 = p_owned_map->getLocalElement(globalNode2);

                        if (isLocalNode1)
                            Kokkos::atomic_fetch_add(&rowCounts(lclNode1), 1);
                        if (isDiffNodes && isLocalNode2)
                            Kokkos::atomic_fetch_add(&rowCounts(lclNode2), 1);
                    }
                }
            }
        });

    Kokkos::View<size_t *> rowOffsets("row offsets", num_rows + 1);

    // Parallel prefix-sum row counts and allocate column index array
    Kokkos::parallel_scan(
        num_rows,
        KOKKOS_LAMBDA(int irow, int &update, bool final) {
            // parallel scan is a multi-pass parallel pattern
            // In the ‘final’ pass ‘update’ has the prefix value
            if (final)
                rowOffsets(irow) = update;
            update += rowCounts(irow);
            if (final && num_rows == irow + 1)
                rowOffsets(irow + 1) = update; // total non-zeros
        });

    Kokkos::deep_copy(rowCounts, static_cast<size_t>(0));
    Kokkos::View<Tpetra_LO *> colIndices("column indices", rowOffsets(num_rows));

    // Fill column index array with rows in non-deterministic order
    Kokkos::parallel_for(
        nodenode.capacity(),
        KOKKOS_LAMBDA(int ientry) {
            if (nodenode.valid_at(ientry))
            {
                const pair<Tpetra_GO, Tpetra_GO> key = nodenode.key_at(ientry);
                const Tpetra_GO globalNode1 = key.first;
                const Tpetra_GO globalNode2 = key.second;

                const bool isDiffNodes = (globalNode1 != globalNode2);
                const bool isLocalNode1 = p_owned_map->isNodeGlobalElement(globalNode1);
                const bool isLocalNode2 = p_owned_map->isNodeGlobalElement(globalNode2);

                const Tpetra_LO lclNode1 = p_owned_map->getLocalElement(globalNode1);
                const Tpetra_LO lclNode2 = p_owned_map->getLocalElement(globalNode2);

                const Tpetra_LO lclNodeWO1 = p_overlapped_map->getLocalElement(globalNode1);
                const Tpetra_LO lclNodeWO2 = p_overlapped_map->getLocalElement(globalNode2);

                if (isLocalNode1)
                {
                    const Tpetra_LO lclRow = lclNode1;
                    const Tpetra_LO lclCol = lclNodeWO2;

                    const size_t count = Kokkos::atomic_fetch_add(&rowCounts(lclRow), 1);
                    colIndices(rowOffsets(lclRow) + count) = lclCol;
                }
                if (isDiffNodes && isLocalNode2)
                {
                    const Tpetra_LO lclRow = lclNode2;
                    const Tpetra_LO lclCol = lclNodeWO1;

                    const size_t count = Kokkos::atomic_fetch_add(&rowCounts(lclRow), 1);
                    colIndices(rowOffsets(lclRow) + count) = lclCol;
                }
            }
        });

    // Sort eacch row of column index array
    Kokkos::parallel_for(
        num_rows,
        KOKKOS_LAMBDA(int lclRow) {
            auto currentColIndices = subview(colIndices, Kokkos::make_pair(rowOffsets(lclRow), rowOffsets(lclRow + 1)));
            Tpetra_LO *const lclColIndsRaw = currentColIndices.data();
            std::sort(lclColIndsRaw, lclColIndsRaw + rowOffsets(lclRow + 1) - rowOffsets(lclRow));
        });

    RCP<Tpetra_CrsGraph> Hgraph = rcp(new Tpetra_CrsGraph(p_owned_map, p_overlapped_map, rowOffsets, colIndices));
    Hgraph->fillComplete(p_owned_map, p_owned_map);
    return Hgraph;
}
