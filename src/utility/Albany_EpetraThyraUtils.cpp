#include "Albany_EpetraThyraUtils.hpp"

#include "Thyra_EpetraThyraWrappers.hpp"

namespace Albany
{

// ============ Epetra->Thyra conversion routines ============ //

Teuchos::RCP<const Thyra_VectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Epetra_Map> map)
{
  Teuchos::RCP<const Thyra_VectorSpace> vs;
  if (!map.is_null()) {
    vs = Thyra::create_VectorSpace(map);
  }

  return vs;
}

Teuchos::RCP<Thyra_Vector> 
createThyraVector (const Teuchos::RCP<Epetra_Vector> v)
{
  Teuchos::RCP<Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::create_Vector(v);
  }

  return v_thyra;
}

Teuchos::RCP<const Thyra_Vector> 
createConstThyraVector (const Teuchos::RCP<const Epetra_Vector> v)
{
  Teuchos::RCP<const Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::create_Vector(v);
  }

  return v_thyra;
}

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Epetra_MultiVector> mv)
{
  Teuchos::RCP<Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::create_MultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Epetra_MultiVector> mv)
{
  Teuchos::RCP<const Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::create_MultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp (const Teuchos::RCP<Epetra_Operator> op)
{
  Teuchos::RCP<Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::nonconstEpetraLinearOp(op);
  }

  return lop;
}

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp (const Teuchos::RCP<const Epetra_Operator> op)
{
  Teuchos::RCP<const Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::epetraLinearOp(op);
  }

  return lop;
}

// ============ Thyra->Epetra conversion routines ============ //

Teuchos::RCP<const Epetra_Map>
getEpetraMap (const Teuchos::RCP<const Thyra_VectorSpace> vs,
              const bool throw_on_failure)
{
  Teuchos::RCP<const Epetra_Map> map;
  if (!vs.is_null()) {
    if (throw_on_failure) {
      map = Thyra::get_Epetra_Map(vs);
    } else {
      try {
        map = Thyra::get_Epetra_Map(vs);
      } catch (...) {
        // Do nothing
      }
    }
  }

  return map;
}

Teuchos::RCP<Epetra_Vector>
getEpetraVector (const Teuchos::RCP<Thyra_Vector> v,
                 const bool throw_on_failure)
{
  Teuchos::RCP<Epetra_Vector> v_epetra;
  if (!v.is_null()) {
    if (throw_on_failure) {
      v_epetra = Thyra::get_Epetra_Vector(v);
    } else {
      try {
        v_epetra = Thyra::get_Epetra_Vector(v);
      } catch (...) {
        // Do nothing
      }
    }
  }

  return v_epetra;
}

Teuchos::RCP<const Epetra_Vector>
getConstEpetraVector (const Teuchos::RCP<const Thyra_Vector> v,
                      const bool throw_on_failure)

{
  Teuchos::RCP<const Epetra_Vector> v_epetra;
  if (!v.is_null()) {
    if (throw_on_failure) {
      v_epetra = Thyra::get_Epetra_Vector(v);
    } else {
      try {
        v_epetra = Thyra::get_Epetra_Vector(v);
      } catch (...) {
        // Do nothing
      }
    }
  }

  return v_epetra;
}

Teuchos::RCP<Epetra_MultiVector>
getEpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv,
                      const bool throw_on_failure)
{
  Teuchos::RCP<Epetra_MultiVector> mv_epetra;
  if (!mv.is_null()) {
    if (throw_on_failure) {
      mv_epetra = Thyra::get_Epetra_MultiVector(mv);
    } else {
      try {
        mv_epetra = Thyra::get_Epetra_MultiVector(mv);
      } catch (...) {
        // Do nothing
      }
    }
  }

  return mv_epetra;
}

Teuchos::RCP<const Epetra_MultiVector>
getConstEpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                           const bool throw_on_failure)
{
  Teuchos::RCP<const Epetra_MultiVector> mv_epetra;
  if (!mv.is_null()) {
    if (throw_on_failure) {
      mv_epetra = Thyra::get_Epetra_MultiVector(mv);
    } else {
      try {
        mv_epetra = Thyra::get_Epetra_MultiVector(mv);
      } catch (...) {
        // Do nothing
      }
    }
  }

  return mv_epetra;
}

Teuchos::RCP<Epetra_Operator>
getEpetraOperator (const Teuchos::RCP<Thyra_LinearOp> lop,
                   const bool throw_on_failure)
{
  Teuchos::RCP<Epetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra::EpetraLinearOp>(lop,throw_on_failure);
    if (!tmp.is_null()) {
      op = tmp->epetra_op();
    }
  }

  return op;
}

Teuchos::RCP<const Epetra_Operator>
getConstEpetraOperator (const Teuchos::RCP<const Thyra_LinearOp> lop,
                        const bool throw_on_failure)
{
  Teuchos::RCP<const Epetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra::EpetraLinearOp>(lop,throw_on_failure);
    if (!tmp.is_null()) {
      op = tmp->epetra_op();
    }
  }

  return op;
}

Teuchos::RCP<Epetra_CrsMatrix>
getEpetraMatrix (const Teuchos::RCP<Thyra_LinearOp> lop,
                 const bool throw_on_failure)
{
  Teuchos::RCP<Epetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getEpetraOperator(lop,throw_on_failure);
    mat = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(op,throw_on_failure);
  }

  return mat;
}

Teuchos::RCP<const Epetra_CrsMatrix>
getConstEpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp> lop,
                      const bool throw_on_failure)
{
  Teuchos::RCP<const Epetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getEpetraOperator(lop,throw_on_failure);
    mat = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(op,throw_on_failure);
  }

  return mat;
}

} // namespace Albany
