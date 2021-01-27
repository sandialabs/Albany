#ifndef ALBANY_HESSIAN_HPP
#define ALBANY_HESSIAN_HPP

namespace Albany
{
    /**
     * \brief createHessianCrsGraph function
     *
     * This function computes the Tpetra::CrsGraph associated to
     * the Hessian w.r.t a distributed parameter.
     *
     * \param p_owned_map [in] Tpetra::Map which specifies the owned entries of the current distributed parameter.
     *
     * \param p_overlapped_map [in] Tpetra::Map which specifies the overlapped entries of the current distributed parameter.
     *
     * \param wsElDofs [in] Vector of IDArray associated to the mesh used.
     */
    Teuchos::RCP<Tpetra_CrsGraph> createHessianCrsGraph(
        Teuchos::RCP<const Tpetra_Map> p_owned_map,
        Teuchos::RCP<const Tpetra_Map> p_overlapped_map,
        const std::vector<IDArray> wsElDofs);

} // namespace Albany

#endif // ALBANY_HESSIAN_HPP
