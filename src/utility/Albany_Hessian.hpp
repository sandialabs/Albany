#ifndef ALBANY_HESSIAN_HPP
#define ALBANY_HESSIAN_HPP

namespace Albany
{
    Teuchos::RCP<Tpetra_CrsGraph> createHessianCrsGraph(
        Teuchos::RCP<const Tpetra_Map> p_owned_map,
        Teuchos::RCP<const Tpetra_Map> p_overlapped_map,
        const std::vector<IDArray> wsElDofs);

} // namespace Albany

#endif // ALBANY_HESSIAN_HPP
