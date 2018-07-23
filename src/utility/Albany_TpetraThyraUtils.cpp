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
getTpetraMap (const Teuchos::RCP<const Thyra_VectorSpace> vs)
{
  Teuchos::RCP<const Tpetra_Map> map;
  if (!vs.is_null()) {
    // There is no way to access the tpetra map in the Thyra_TpetraVectorSpace class,
    // so create a vector, grab the Tpetra_Vector, then grab the map from it
    auto v = Thyra::createMember(vs);
    map = getConstTpetraVector(v)->getMap();
  }

  return map;
}

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (const Teuchos::RCP<Thyra_Vector> v)
{
  Teuchos::RCP<Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    v_tpetra = ConverterT::getTpetraVector(v);
  }

  return v_tpetra;
}

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector> v)
{
  Teuchos::RCP<const Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    v_tpetra = ConverterT::getConstTpetraVector(v);
  }

  return v_tpetra;
}

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv)
{
  Teuchos::RCP<Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    mv_tpetra = ConverterT::getTpetraMultiVector(mv);
  }

  return mv_tpetra;
}

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv)
{
  Teuchos::RCP<const Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    mv_tpetra = ConverterT::getConstTpetraMultiVector(mv);
  }

  return mv_tpetra;
}

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator (const Teuchos::RCP<Thyra_LinearOp> lop)
{
  Teuchos::RCP<Tpetra_Operator> op;
  if (!lop.is_null()) {
    op = ConverterT::getTpetraOperator(lop);
  }

  return op;
}

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator (const Teuchos::RCP<const Thyra_LinearOp> lop)
{
  Teuchos::RCP<const Tpetra_Operator> op;
  if (!lop.is_null()) {
    op = ConverterT::getConstTpetraOperator(lop);
  }

  return op;
}

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix (const Teuchos::RCP<Thyra_LinearOp> lop)
{
  Teuchos::RCP<Tpetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getTpetraOperator(lop);
    mat = Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(op);
    TEUCHOS_TEST_FOR_EXCEPTION(mat.is_null(), std::runtime_error, "Error! Could not cast Tpetra_Operator to Tpetra_CrsMatrix.\n");
  }

  return mat;
}

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp> lop)
{
  Teuchos::RCP<const Tpetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getTpetraOperator(lop);
    mat = Teuchos::rcp_dynamic_cast<const Tpetra_CrsMatrix>(op);
    TEUCHOS_TEST_FOR_EXCEPTION(mat.is_null(), std::runtime_error, "Error! Could not cast Tpetra_Operator to Tpetra_CrsMatrix.\n");
  }

  return mat;
}

} // namespace Albany
