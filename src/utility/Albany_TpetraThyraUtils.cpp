#include "Albany_TpetraThyraUtils.hpp"

#include "Albany_TpetraThyraTypes.hpp"

namespace Albany
{

struct BadThyraTpetraCast : public std::bad_cast {
  BadThyraTpetraCast (const std::string& msg)
   : m_msg (msg)
  {}

  const char * what () const noexcept { return m_msg.c_str(); }

private:
  const std::string& m_msg;
};

// ============ Tpetra->Thyra conversion routines ============ //

Teuchos::RCP<const Thyra_SpmdVectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Tpetra_Map>& map)
{
  Teuchos::RCP<const Thyra_SpmdVectorSpace> vs;
  if (!map.is_null()) {
    vs = Thyra::tpetraVectorSpace<ST>(map);
  }

  return vs;
}

Teuchos::RCP<Thyra_Vector>createThyraVector (const Teuchos::RCP<Tpetra_Vector>& v)
{
  Teuchos::RCP<Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::createVector(v);
  }

  return v_thyra;
}

Teuchos::RCP<const Thyra_Vector>createConstThyraVector (const Teuchos::RCP<const Tpetra_Vector>& v)
{
  Teuchos::RCP<const Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::createConstVector(v);
  }

  return v_thyra;
}

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Tpetra_MultiVector>& mv)
{
  Teuchos::RCP<Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::createMultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Tpetra_MultiVector>& mv)
{
  Teuchos::RCP<const Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::createConstMultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp (const Teuchos::RCP<Tpetra_Operator>& op)
{
  Teuchos::RCP<Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::createLinearOp(op);
  }

  return lop;
}

Teuchos::RCP<Thyra_BlockedLinearOp>
createThyraBlockedLinearOp (const Teuchos::RCP<Tpetra_Operator>& op)
{
  Teuchos::RCP<Thyra_BlockedLinearOp> lop;
// GAH fill in functionality here
//  if (!op.is_null()) {
//    lop = Thyra::createLinearOp(op);
//  }

  return lop;
}

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp (const Teuchos::RCP<const Tpetra_Operator>& op)
{
  Teuchos::RCP<const Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::createConstLinearOp(op);
  }

  return lop;
}

// ============ Thyra->Tpetra conversion routines ============ //

Teuchos::RCP<const Tpetra_Map>
getTpetraMap (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
              const bool throw_if_not_tpetra)
{
  Teuchos::RCP<const Tpetra_Map> map;
  if (!vs.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraVectorSpace>(vs,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      map = tmp->getTpetraMap();
    }
  }

  return map;
}

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (const Teuchos::RCP<Thyra_Vector>& v,
                 const bool throw_if_not_tpetra)
{
  Teuchos::RCP<Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra_TpetraVector>(v,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      v_tpetra = tmp->getTpetraVector();
    }
  }

  return v_tpetra;
}

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector>& v,
                      const bool throw_if_not_tpetra)

{
  Teuchos::RCP<const Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraVector>(v,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      v_tpetra = tmp->getConstTpetraVector();
    }
  }

  return v_tpetra;
}

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector>& mv,
                      const bool throw_if_not_tpetra)
{
  Teuchos::RCP<Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra_TpetraMultiVector>(mv,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      mv_tpetra = tmp->getTpetraMultiVector();
    }
  }

  return mv_tpetra;
}

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector>& mv,
                           const bool throw_if_not_tpetra)
{
  Teuchos::RCP<const Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraMultiVector>(mv,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      mv_tpetra = tmp->getConstTpetraMultiVector();
    }
  }

  return mv_tpetra;
}

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator (const Teuchos::RCP<Thyra_LinearOp>& lop,
                   const bool throw_if_not_tpetra)
{
  Teuchos::RCP<Tpetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra_TpetraLinearOp>(lop,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      op = tmp->getTpetraOperator();
    }
  }

  return op;
}

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                        const bool throw_if_not_tpetra)
{
  Teuchos::RCP<const Tpetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra_TpetraLinearOp>(lop,throw_if_not_tpetra);
    if (!tmp.is_null()) {
      op = tmp->getConstTpetraOperator();
    }
  }

  return op;
}

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix (const Teuchos::RCP<Thyra_LinearOp>& lop,
                 const bool throw_if_not_tpetra)
{
  Teuchos::RCP<Tpetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getTpetraOperator(lop,throw_if_not_tpetra);
    mat = Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(op,throw_if_not_tpetra);
  }

  return mat;
}

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      const bool throw_if_not_tpetra)
{
  Teuchos::RCP<const Tpetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getConstTpetraOperator(lop,throw_if_not_tpetra);
    mat = Teuchos::rcp_dynamic_cast<const Tpetra_CrsMatrix>(op,throw_if_not_tpetra);
  }

  return mat;
}

// --- Casts taking references as inputs --- //

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (Thyra_Vector& v,
                 const bool throw_if_not_tpetra)
{
  Thyra_TpetraVector* tv = dynamic_cast<Thyra_TpetraVector*>(&v);

  if (tv==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_tpetra, BadThyraTpetraCast,
                               "Error! Could not cast input Thyra_Vector to Thyra_TpetraVector.\n");

    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(tv->getTpetraVector().is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraVector object stores a null pointer.\n");

    return tv->getTpetraVector();
  }
}

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Thyra_Vector& v,
                      const bool throw_if_not_tpetra)
{
  const Thyra_TpetraVector* tv = dynamic_cast<const Thyra_TpetraVector*>(&v);

  if (tv==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_tpetra, BadThyraTpetraCast,
                               "Error! Could not cast input Thyra_Vector to Thyra_TpetraVector.\n");
    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(tv->getConstTpetraVector().is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraVector object stores a null pointer.\n");

    return tv->getConstTpetraVector();
  }
}

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (Thyra_MultiVector& mv,
                      const bool throw_if_not_tpetra)
{
  Thyra_TpetraMultiVector* tmv = dynamic_cast<Thyra_TpetraMultiVector*>(&mv);

  if (tmv==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_tpetra, BadThyraTpetraCast,
                               "Error! Could not cast input Thyra_MultiVector to Thyra_TpetraMultiVector.\n");
    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(tmv->getTpetraMultiVector().is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraMultiVector object stores a null pointer.\n");
    return tmv->getTpetraMultiVector();
  }
}

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Thyra_MultiVector& mv,
                           const bool throw_if_not_tpetra)
{
  const Thyra_TpetraMultiVector* tmv = dynamic_cast<const Thyra_TpetraMultiVector*>(&mv);

  if (tmv==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_tpetra, BadThyraTpetraCast,
                               "Error! Could not cast input Thyra_MultiVector to Thyra_TpetraMultiVector.\n");
    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(tmv->getConstTpetraMultiVector().is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraMultiVector object stores a null pointer.\n");
    return tmv->getConstTpetraMultiVector();
  }
}

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator (Thyra_LinearOp& lop,
                   const bool throw_if_not_tpetra)
{
  Thyra_TpetraLinearOp* top = dynamic_cast<Thyra_TpetraLinearOp*>(&lop);

  if (top==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_tpetra, BadThyraTpetraCast,
                               "Error! Could not cast input Thyra_LinearOp to Thyra_TpetraLinearOp.\n");
    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(top->getTpetraOperator().is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraLinearOp object stores a null pointer.\n");
    return top->getTpetraOperator();
  }
}

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator (const Thyra_LinearOp& lop,
                        const bool throw_if_not_tpetra)
{
  const Thyra_TpetraLinearOp* top = dynamic_cast<const Thyra_TpetraLinearOp*>(&lop);

  if (top==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_tpetra, BadThyraTpetraCast,
                               "Error! Could not cast input Thyra_LinearOp to Thyra_TpetraLinearOp.\n");
    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(top->getConstTpetraOperator().is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraLinearOp object stores a null pointer.\n");
    return top->getConstTpetraOperator();
  }
}

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix (Thyra_LinearOp& lop,
                 const bool throw_if_not_tpetra)
{
  Teuchos::RCP<Tpetra_Operator> top = getTpetraOperator(lop,throw_if_not_tpetra);

  if (!top.is_null()) {
    // We allow bad cast, but once cast goes through, we *expect* the operator to store a crs matrix
    auto tmat = Teuchos::rcp_dynamic_cast<Tpetra_CrsMatrix>(top);
    TEUCHOS_TEST_FOR_EXCEPTION(tmat.is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraLinearOp object does not store a Tpetra_CrsMatrix.\n");
    return tmat;
  }
  return Teuchos::null;
}

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix (const Thyra_LinearOp& lop,
                      const bool throw_if_not_tpetra)
{
  Teuchos::RCP<const Tpetra_Operator> top = getConstTpetraOperator(lop,throw_if_not_tpetra);

  if (!top.is_null()) {
    // We allow bad cast, but once cast goes through, we *expect* the operator to store a crs matrix
    auto tmat = Teuchos::rcp_dynamic_cast<const Tpetra_CrsMatrix>(top);
    TEUCHOS_TEST_FOR_EXCEPTION(tmat.is_null(), std::runtime_error,
                               "Error! The Thyra_TpetraLinearOp object does not store a Tpetra_CrsMatrix.\n");
    return tmat;
  }
  return Teuchos::null;
}

} // namespace Albany
