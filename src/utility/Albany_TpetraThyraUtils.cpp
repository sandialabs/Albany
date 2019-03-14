#include "Albany_TpetraThyraUtils.hpp"

#include "Albany_TpetraThyraTypes.hpp"

namespace Albany
{

// ============ Tpetra->Thyra conversion routines ============ //

Teuchos::RCP<const Thyra_VectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Tpetra_Map> map)
{
  Teuchos::RCP<const Thyra_VectorSpace> vs;
  if (!map.is_null()) {
    vs = Thyra::tpetraVectorSpace<ST>(map);
  }

  return vs;
}

Teuchos::RCP<Thyra_Vector> 
createThyraVector (const Teuchos::RCP<Tpetra_Vector> v)
{
  Teuchos::RCP<Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::createVector(v);
  }

  return v_thyra;
}

Teuchos::RCP<const Thyra_Vector> 
createConstThyraVector (const Teuchos::RCP<const Tpetra_Vector> v)
{
  Teuchos::RCP<const Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::createConstVector(v);
  }

  return v_thyra;
}

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Tpetra_MultiVector> mv)
{
  Teuchos::RCP<Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::createMultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Tpetra_MultiVector> mv)
{
  Teuchos::RCP<const Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::createConstMultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp (const Teuchos::RCP<Tpetra_Operator> op)
{
  Teuchos::RCP<Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::createLinearOp(op);
  }

  return lop;
}

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp (const Teuchos::RCP<const Tpetra_Operator> op)
{
  Teuchos::RCP<const Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::createConstLinearOp(op);
  }

  return lop;
}

// ============ Thyra->Tpetra conversion routines ============ //

Teuchos::RCP<const Tpetra_Map>
getTpetraMap (const Teuchos::RCP<const Thyra_VectorSpace> vs,
              const bool throw_on_failure)
{
  Teuchos::RCP<const Tpetra_Map> map;
  if (!vs.is_null()) {
    // There is no way to access the tpetra map in the Thyra_TpetraVectorSpace class,
    // so create a vector, grab the Tpetra_Vector, then grab the map from it
    auto v = Thyra::createMember(vs);
    map = getConstTpetraVector(v,throw_on_failure)->getMap();
  }

  return map;
}

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (const Teuchos::RCP<Thyra_Vector> v,
                 const bool throw_on_failure)
{
  Teuchos::RCP<Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra_TpetraVector>(v,throw_on_failure);
    if (!tmp.is_null()) {
      v_tpetra = tmp->getTpetraVector();
    }
  }

  return v_tpetra;
}

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector> v,
                      const bool throw_on_failure)

{
  Teuchos::RCP<const Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraVector>(v,throw_on_failure);
    if (!tmp.is_null()) {
      v_tpetra = tmp->getConstTpetraVector();
    }
  }

  return v_tpetra;
}

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv,
                      const bool throw_on_failure)
{
  Teuchos::RCP<Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra_TpetraMultiVector>(mv,throw_on_failure);
    if (!tmp.is_null()) {
      mv_tpetra = tmp->getTpetraMultiVector();
    }
  }

  return mv_tpetra;
}

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                           const bool throw_on_failure)
{
  Teuchos::RCP<const Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraMultiVector>(mv,throw_on_failure);
    if (!tmp.is_null()) {
      mv_tpetra = tmp->getConstTpetraMultiVector();
    }
  }

  return mv_tpetra;
}

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator (const Teuchos::RCP<Thyra_LinearOp> lop,
                   const bool throw_on_failure)
{
  Teuchos::RCP<Tpetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra_TpetraLinearOp>(lop,throw_on_failure);
    if (!tmp.is_null()) {
      op = tmp->getTpetraOperator();
    }
  }

  return op;
}

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator (const Teuchos::RCP<const Thyra_LinearOp> lop,
                        const bool throw_on_failure)
{
  Teuchos::RCP<const Tpetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraLinearOp>(lop,throw_on_failure);
    if (!tmp.is_null()) {
      op = tmp->getConstTpetraOperator();
    }
  }

  return op;
}

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix (const Teuchos::RCP<Thyra_LinearOp> lop,
                 const bool throw_on_failure)
{
  Teuchos::RCP<Tpetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getTpetraOperator(lop,throw_on_failure);
    mat = Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(op,throw_on_failure);
  }

  return mat;
}

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp> lop,
                      const bool throw_on_failure)
{
  Teuchos::RCP<const Tpetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getConstTpetraOperator(lop,throw_on_failure);
    mat = Teuchos::rcp_dynamic_cast<const Tpetra_CrsMatrix>(op,throw_on_failure);
  }

  return mat;
}

} // namespace Albany
