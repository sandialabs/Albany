#include "Albany_EpetraThyraUtils.hpp"
#include "Albany_CommTypes.hpp"
#include "Albany_Macros.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Epetra_LocalMap.h"

// We only use Thyra's Epetra wrapper for the linear op.
// For vec/mvec, we rely on spmd interface
#include "Thyra_EpetraLinearOp.hpp"

#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSpmdMultiVector.hpp"
#include "Thyra_DefaultSpmdVector.hpp"
#include "Thyra_ScalarProdVectorSpaceBase.hpp"

namespace Albany
{

struct BadThyraEpetraCast : public std::bad_cast {
  BadThyraEpetraCast (const std::string& msg)
   : m_msg (msg)
  {}

  const char * what () const noexcept { return m_msg.c_str(); }

private:
  const std::string& m_msg;
};

// ============ Epetra->Thyra conversion routines ============ //

Teuchos::RCP<const Thyra_SpmdVectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Epetra_BlockMap> bmap)
{
  Teuchos::RCP<const Thyra_SpmdVectorSpace> vs;
  if (!bmap.is_null()) {

    auto comm = createTeuchosCommFromEpetraComm(bmap->Comm());
    vs = Thyra::defaultSpmdVectorSpace<ST>(createThyraCommFromTeuchosComm(comm), bmap->NumMyElements(), bmap->NumGlobalElements64(), !bmap->DistributedGlobal());

    // Attach the bmap to the new RCP, so it doesn't get destroyed as long as the new vs lives
    // Note: if the input is a weak rcp, this may not work.
    Teuchos::set_extra_data(bmap, "Epetra_BlockMap", inoutArg(vs) );
  }

  return vs;
}


Teuchos::RCP<Thyra_Vector> 
createThyraVector (const Teuchos::RCP<Epetra_Vector>& v)
{
  Teuchos::RCP<Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    auto vs = createThyraVectorSpace(Teuchos::rcpFromRef(v->Map()));

    Teuchos::ArrayRCP<ST> vals(v->Values(),0,v->MyLength(),false);
    v_thyra = Teuchos::rcp( new Thyra::DefaultSpmdVector<ST>(vs,vals,1) );

    // Attach the input vector to the new RCP, so it doesn't get destroyed as long as the new vector lives
    // Note: if the input is a weak rcp, this may not work.
    Teuchos::set_extra_data(v, "Epetra_Vector", inoutArg(v_thyra));
  }

  return v_thyra;
}

Teuchos::RCP<const Thyra_Vector> 
createConstThyraVector (const Teuchos::RCP<const Epetra_Vector>& v)
{
  Teuchos::RCP<const Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    auto vs = createThyraVectorSpace(Teuchos::rcpFromRef(v->Map()));
    Teuchos::ArrayRCP<ST> vals(v->Values(),0,v->MyLength(),false);
    v_thyra = Teuchos::rcp( new Thyra::DefaultSpmdVector<ST>(vs,vals,1) );

    // Attach the input vector to the new RCP, so it doesn't get destroyed as long as the new vector lives
    // Note: if the input is a weak rcp, this may not work.
    Teuchos::set_extra_data(v, "Epetra_Vector", inoutArg(v_thyra));
  }

  return v_thyra;
}

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Epetra_MultiVector>& mv)
{
  Teuchos::RCP<Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    Teuchos::RCP<const Thyra_SpmdVectorSpace> range  = createThyraVectorSpace(Teuchos::rcpFromRef(mv->Map()));
    // LB: I have NO IDEA why the rcp_implicit_cast is needed (RCP should already be polymorphic). Yet, the compiler complains without it.
    Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<ST>> domain =
      Thyra::createSmallScalarProdVectorSpaceBase(Teuchos::rcp_implicit_cast<const Thyra_VectorSpace>(range),mv->NumVectors());

    Teuchos::ArrayRCP<ST> vals(mv->Values(),0,mv->MyLength()*mv->NumVectors(),false);
    mv_thyra = Teuchos::rcp(new Thyra::DefaultSpmdMultiVector<ST>(range,domain,vals));

    // Attach the input mv to the new RCP, so it doesn't get destroyed as long as the new mv lives
    // Note: if the input is a weak rcp, this may not work.
    Teuchos::set_extra_data(mv, "Epetra_MultiVector", inoutArg(mv_thyra));
  }

  return mv_thyra;
}

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Epetra_MultiVector>& mv)
{
  Teuchos::RCP<const Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    Teuchos::RCP<const Thyra_SpmdVectorSpace> range  = createThyraVectorSpace(Teuchos::rcpFromRef(mv->Map()));
    // LB: I have NO IDEA why the rcp_implicit_cast is needed (RCP should already be polymorphic). Yet, the compiler complains without it.
    Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<ST>> domain =
      Thyra::createSmallScalarProdVectorSpaceBase(Teuchos::rcp_implicit_cast<const Thyra_VectorSpace>(range),mv->NumVectors());

    Teuchos::ArrayRCP<ST> vals(mv->Values(),0,mv->MyLength()*mv->NumVectors(),false);
    mv_thyra = Teuchos::rcp( new Thyra::DefaultSpmdMultiVector<ST>(range,domain,vals) );

    // Attach the input mv to the new RCP, so it doesn't get destroyed as long as the new mv lives
    // Note: if the input is a weak rcp, this may not work.
    Teuchos::set_extra_data(mv, "Epetra_MultiVector", inoutArg(mv_thyra));
  }

  return mv_thyra;
}

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp (const Teuchos::RCP<Epetra_Operator>& op)
{
  Teuchos::RCP<Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::nonconstEpetraLinearOp(op);
  }

  return lop;
}

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp (const Teuchos::RCP<const Epetra_Operator>& op)
{
  Teuchos::RCP<const Thyra_LinearOp> lop;
  if (!op.is_null()) {
    lop = Thyra::epetraLinearOp(op);
  }

  return lop;
}

// ============ Thyra->Epetra conversion routines ============ //

Teuchos::RCP<const Epetra_BlockMap>
getEpetraBlockMap (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                   const bool throw_if_not_epetra)
{
  Teuchos::RCP<const Epetra_BlockMap> map;
  if (!vs.is_null()) {
    // Note: in the Epetra case, we should always build our initial vector spaces from
    //       the utility createThyraVectorSpace in this file. Such vs always contain
    //       extra data with the original map. All the LinearOp objects (and its derived
    //       types) that we create from such vs should contain a copy of the vector space.
    //       This means that *all* the vector spaces corresponding to Epetra as a backend
    //       should *always* contain extra data.
    //       There are two exceptions:
    //        1) vectors created from the domain of a multivector, i.e., from a vs generated
    //           by a small vs factory. In this case, we should be able to cast to a
    //           DefaultSpmdVectorSpace, AND the vs should be locally replicated.
    //        2) vectors created from a vs that was created from an Epetra_Map inside trilinos.
    //           Trilinos would use the create_VectorSpace routine from ThyraEpetraAdapters,
    //           which attaches an RCP<const Epetra_Map> to the generated vs RCP.

    auto data  = Teuchos::get_optional_extra_data<Teuchos::RCP<const Epetra_BlockMap>>(vs,"Epetra_BlockMap");
    auto data2 = Teuchos::get_optional_extra_data<Teuchos::RCP<const Epetra_Map>>(vs,"epetra_map");
    if (!data.is_null()) {
      map = *data;
    } else if (!data2.is_null()) {
      map = *data2;
    } else {
      auto spmd_vs = getSpmdVectorSpace(vs);
      TEUCHOS_TEST_FOR_EXCEPTION (spmd_vs.is_null() && throw_if_not_epetra, std::runtime_error,
                                  "Error! Could not extract/build Epetra_BlockMap from Thyra_VectorSpace.\n");

      const bool isLocallyReplicated = spmd_vs->isLocallyReplicated();
      TEUCHOS_TEST_FOR_EXCEPTION (!isLocallyReplicated, std::logic_error,
                                  "Error! The input map is convertible to a SpmdVectorSpaceBase, but it is not locally replicated.\n"
                                  "       This should not happen. Please, contact developers.\n");

      auto t_comm = createTeuchosCommFromThyraComm(spmd_vs->getComm());
      auto e_comm = createEpetraCommFromTeuchosComm(t_comm);
      map = Teuchos::rcp( new Epetra_LocalMap(static_cast<int>(spmd_vs->localSubDim()),0,*e_comm) );
    }
  }

  return map;
}

Teuchos::RCP<const Epetra_Map>
getEpetraMap (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
              const bool throw_if_not_epetra)
{
  Teuchos::RCP<const Epetra_BlockMap> bmap = getEpetraBlockMap(vs,throw_if_not_epetra);

  // If we are failure-tolerant, if the call failed, we must exit now
  if (!throw_if_not_epetra && bmap.is_null()) {
    return Teuchos::null;
  }

  const Epetra_Map* raw_map = reinterpret_cast<const Epetra_Map*>(bmap.get());

  Teuchos::RCP<const Epetra_Map> map(raw_map,bmap.access_private_node());

  return map;
}

Teuchos::RCP<Epetra_Vector>
getEpetraVector (const Teuchos::RCP<Thyra_Vector>& v,
                 const bool throw_if_not_epetra)
{
  Teuchos::RCP<Epetra_Vector> v_epetra;
  if (!v.is_null()) {
    auto data = Teuchos::get_optional_extra_data<Teuchos::RCP<Epetra_Vector>>(v,"Epetra_Vector");

    if (!data.is_null()) {
      v_epetra = *data;
    } else {
      auto spmd_v = Teuchos::rcp_dynamic_cast<Thyra::DefaultSpmdVector<ST>>(v);
      if (spmd_v.is_null()) {
        // Give up
        TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                                   "Error! Could not cast input Thyra_Vector to Thyra::DefaultSpmdVector<ST>.\n");
      } else {
        // Get the map, then create a vector from the spmd_v values and the map
        auto emap = getEpetraMap(v->space());
        Teuchos::ArrayRCP<ST> vals = spmd_v->getRCPtr();

        v_epetra = Teuchos::rcp(new Epetra_Vector(View,*emap,vals.getRawPtr()));
        // Attach the input vector to the newly created one, to prolong its life
        // Note: if the input is a weak rcp, this may not work.
        Teuchos::set_extra_data(v, "values_arcp", inoutArg(v_epetra) );
      }
    }
  }

  return v_epetra;
}

Teuchos::RCP<const Epetra_Vector>
getConstEpetraVector (const Teuchos::RCP<const Thyra_Vector>& v,
                      const bool throw_if_not_epetra)

{
  Teuchos::RCP<const Epetra_Vector> v_epetra;
  if (!v.is_null()) {
    // The thyra vector may have been originally created from a non-const Epetra_Vector,
    // so we need to check both const and nonconst
    auto data_const    = Teuchos::get_optional_extra_data<Teuchos::RCP<const Epetra_Vector>>(v,"Epetra_Vector");
    auto data_nonconst = Teuchos::get_optional_extra_data<Teuchos::RCP<Epetra_Vector>>(v,"Epetra_Vector");

    if (!data_const.is_null()) {
      v_epetra = *data_const;
    } else if (!data_nonconst.is_null()) {
      v_epetra = *data_nonconst;
    } else {
      auto spmd_v = Teuchos::rcp_dynamic_cast<const Thyra::DefaultSpmdVector<ST>>(v);
      if (spmd_v.is_null()) {
        // Give up
        TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                                   "Error! Could not cast input Thyra_Vector to Thyra::DefaultSpmdVector<ST>.\n");
      } else {
        // Get the map, then create a vector from the spmd_v values and the map
        auto emap = getEpetraMap(v->space());
        Teuchos::ArrayRCP<const ST> vals = spmd_v->getRCPtr();

        // Unfortunately, the constructor for Epetra_Vector takes a double* rather than a const double* (rightfully so),
        // so there's really no way around the const cast. It is innocuous and without side effects though, since we
        // are going to create an RCP<const Epetra_Vector>.
        ST* vals_nonconst = const_cast<ST*>(vals.getRawPtr());
        v_epetra = Teuchos::rcp(new Epetra_Vector(View,*emap,vals_nonconst));

        // Attach the input vector to the newly created one, to prolong its life
        // Note: if the input is a weak rcp, this may not work.
        Teuchos::set_extra_data(v, "values_arcp", inoutArg(v_epetra) );
      }
    }
  }

  return v_epetra;
}

Teuchos::RCP<Epetra_MultiVector>
getEpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector>& mv,
                      const bool throw_if_not_epetra)
{
  Teuchos::RCP<Epetra_MultiVector> mv_epetra;
  if (!mv.is_null()) {
    auto data = Teuchos::get_optional_extra_data<Teuchos::RCP<Epetra_MultiVector>>(mv,"Epetra_MultiVector");

    if (!data.is_null()) {
      mv_epetra = *data;
    } else {
      auto spmd_mv = Teuchos::rcp_dynamic_cast<Thyra::DefaultSpmdMultiVector<ST>>(mv);
      if (spmd_mv.is_null()) {
        // Give up
        TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                                   "Error! Could not cast input Thyra_MultiVector to Thyra::DefaultSpmdMultiVector<ST>.\n");
      } else {
        // Get the map, then create a vector from the spmd_v values and the map
        auto emap = getEpetraMap(mv->range());

        Teuchos::ArrayRCP<ST> vals;
        Teuchos::Ordinal leadingDim;
        spmd_mv->getNonconstLocalData(Teuchos::outArg(vals),Teuchos::outArg(leadingDim));

        mv_epetra = Teuchos::rcp(new Epetra_MultiVector(View,*emap,vals.getRawPtr(),static_cast<int>(leadingDim),mv->domain()->dim()));

        // Attach the input mv to the newly created one, to prolong its life
        // Note: if the input is a weak rcp, this may not work.
        Teuchos::set_extra_data(mv, "values_arcp", inoutArg(mv_epetra) );
      }
    }
  }

  return mv_epetra;
}

Teuchos::RCP<const Epetra_MultiVector>
getConstEpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector>& mv,
                           const bool throw_if_not_epetra)
{
  Teuchos::RCP<const Epetra_MultiVector> mv_epetra;
  if (!mv.is_null()) {
    // The thyra vector may have been originally created from a non-const Epetra_Vector,
    // so we need to check both const and nonconst
    auto data_const    = Teuchos::get_optional_extra_data<Teuchos::RCP<const Epetra_MultiVector>>(mv,"Epetra_MVector");
    auto data_nonconst = Teuchos::get_optional_extra_data<Teuchos::RCP<Epetra_MultiVector>>(mv,"Epetra_MultiVector");

    TEUCHOS_TEST_FOR_EXCEPTION (throw_if_not_epetra && data_const.is_null() && data_nonconst.is_null(), std::runtime_error,
                                "Error! Could not extract Epetra_MultiVector from Thyra_MultiVector.\n");
    if (!data_const.is_null()) {
      mv_epetra = *data_const;
    } else if (!data_nonconst.is_null()) {
      mv_epetra = *data_nonconst;
    } else {
      auto spmd_mv = Teuchos::rcp_dynamic_cast<const Thyra::DefaultSpmdMultiVector<ST>>(mv);
      if (spmd_mv.is_null()) {
        // Give up
        TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                                   "Error! Could not cast input Thyra_MultiVector to Thyra::DefaultSpmdMultiVector<ST>.\n");
      } else {
        // Get the map, then create a vector from the spmd_v values and the map
        auto emap = getEpetraMap(mv->range());

        Teuchos::ArrayRCP<const ST> vals;
        Teuchos::Ordinal leadingDim;
        spmd_mv->getLocalData(Teuchos::outArg(vals),Teuchos::outArg(leadingDim));

        // Unfortunately, the constructor for Epetra_Vector takes a double* rather than a const double* (rightfully so),
        // so there's really no way around the const cast. It is innocuous and without side effects though, since we
        // are going to create an RCP<const Epetra_Vector>.
        ST* vals_nonconst = const_cast<ST*>(vals.getRawPtr());
        mv_epetra = Teuchos::rcp(new Epetra_MultiVector(View,*emap,vals_nonconst,static_cast<int>(leadingDim),mv->domain()->dim()));

        // Attach the input mv to the newly created one, to prolong its life
        // Note: if the input is a weak rcp, this may not work.
        Teuchos::set_extra_data(mv, "values_arcp", inoutArg(mv_epetra) );
      }
    }
  }

  return mv_epetra;
}

Teuchos::RCP<Epetra_Operator>
getEpetraOperator (const Teuchos::RCP<Thyra_LinearOp>& lop,
                   const bool throw_if_not_epetra)
{
  Teuchos::RCP<Epetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<Thyra::EpetraLinearOp>(lop,throw_if_not_epetra);
    if (!tmp.is_null()) {
      op = tmp->epetra_op();
    }
  }
  return op;
}

Teuchos::RCP<const Epetra_Operator>
getConstEpetraOperator (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                        const bool throw_if_not_epetra)
{
  Teuchos::RCP<const Epetra_Operator> op;
  if (!lop.is_null()) {
    auto tmp = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(lop,throw_if_not_epetra);
    if (!tmp.is_null()) {
      op = tmp->epetra_op();
    }
  }
  return op;
}

Teuchos::RCP<Epetra_CrsMatrix>
getEpetraMatrix (const Teuchos::RCP<Thyra_LinearOp>& lop,
                 const bool throw_if_not_epetra)
{
  Teuchos::RCP<Epetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getEpetraOperator(lop,throw_if_not_epetra);
    mat = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(op,throw_if_not_epetra);
  }

  return mat;
}

Teuchos::RCP<const Epetra_CrsMatrix>
getConstEpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      const bool throw_if_not_epetra)
{
  Teuchos::RCP<const Epetra_CrsMatrix> mat;
  if (!lop.is_null()) {
    auto op = getConstEpetraOperator(lop,throw_if_not_epetra);
    mat = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(op,throw_if_not_epetra);
  }

  return mat;
}

// --- Casts taking references as inputs --- //

Teuchos::RCP<Epetra_Vector>
getEpetraVector (Thyra_Vector& v,
                 const bool throw_if_not_epetra)
{
  auto* spmd_v = dynamic_cast<Thyra::DefaultSpmdVector<ST>*>(&v);
  Teuchos::RCP<Epetra_Vector> e_v;
  if (spmd_v==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                               "Error! Could not cast input Thyra_Vector to Thyra::DefaultSpmdVector<ST>.\n");
  } else {
    Teuchos::ArrayRCP<ST> vals = spmd_v->getRCPtr();
    auto emap = getEpetraMap(v.space());

    e_v = Teuchos::rcp(new Epetra_Vector(View,*emap,vals.getRawPtr()));

    // Attach the input vector to the newly created one, to prolong its life
    // Note: the input ref must outlive the output of this routine
    Teuchos::set_extra_data(Teuchos::rcpFromRef(v), "original vector", inoutArg(e_v) );
  }
  return e_v;
}

Teuchos::RCP<const Epetra_Vector>
getConstEpetraVector (const Thyra_Vector& v,
                      const bool throw_if_not_epetra)
{
  auto* spmd_v = dynamic_cast<const Thyra::DefaultSpmdVector<ST>*>(&v);
  Teuchos::RCP<const Epetra_Vector> e_v;
  if (spmd_v==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                               "Error! Could not cast input Thyra_Vector to Thyra::DefaultSpmdVector<ST>.\n");
  } else {
    Teuchos::ArrayRCP<const ST> vals = spmd_v->getRCPtr();
    auto emap = getEpetraMap(v.space());

    // Unfortunately, the constructor for Epetra_Vector takes a double* rather than a const double* (rightfully so),
    // so there's really no way around the const cast. It is innocuous and without side effects though, since we
    // are going to create an RCP<const Epetra_Vector>.
    e_v = Teuchos::rcp(new Epetra_Vector(View,*emap,const_cast<ST*>(vals.getRawPtr())));

    // Attach the input vector to the newly created one, to prolong its life
    // Note: the input ref must outlive the output of this routine
    Teuchos::set_extra_data(Teuchos::rcpFromRef(v), "original vector", inoutArg(e_v) );
  }
  return e_v;
}

Teuchos::RCP<Epetra_MultiVector>
getEpetraMultiVector (Thyra_MultiVector& mv,
                      const bool throw_if_not_epetra)
{
  auto* spmd_mv = dynamic_cast<Thyra::DefaultSpmdMultiVector<ST>*>(&mv);
  Teuchos::RCP<Epetra_MultiVector> e_mv;
  if (spmd_mv==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                               "Error! Could not cast input Thyra_MultiVector to Thyra::DefaultSpmdMultiVector<ST>.\n");
  } else {
    Teuchos::ArrayRCP<ST> vals;
    Teuchos::Ordinal leadingDim;
    spmd_mv->getNonconstLocalData(Teuchos::inOutArg(vals),Teuchos::inOutArg(leadingDim));
    auto emap = getEpetraMap(mv.range());

    e_mv = Teuchos::rcp(new Epetra_MultiVector(View,*emap,vals.get(),leadingDim,mv.domain()->dim()));

    // Attach the input mv to the newly created one, to prolong its life
    // Note: the input ref must outlive the output of this routine
    Teuchos::set_extra_data(Teuchos::rcpFromRef(mv), "original multivector", inoutArg(e_mv) );
  }
  return e_mv;
}

Teuchos::RCP<const Epetra_MultiVector>
getConstEpetraMultiVector (const Thyra_MultiVector& mv,
                           const bool throw_if_not_epetra)
{
  auto* spmd_mv = dynamic_cast<const Thyra::DefaultSpmdMultiVector<ST>*>(&mv);
  Teuchos::RCP<Epetra_MultiVector> e_mv;
  if (spmd_mv==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                               "Error! Could not cast input Thyra_MultiVector to Thyra::DefaultSpmdMultiVector<ST>.\n");
  } else {
    Teuchos::ArrayRCP<const ST> vals;
    Teuchos::Ordinal leadingDim;
    spmd_mv->getLocalData(Teuchos::inOutArg(vals),Teuchos::inOutArg(leadingDim));
    auto emap = getEpetraMap(mv.range());

    // Unfortunately, the constructor for Epetra_MultiVector takes a double* rather than a const double* (rightfully so),
    // so there's really no way around the const cast. It is innocuous and without side effects though, since we
    // are going to create an RCP<const Epetra_MultiVector>.
    e_mv = Teuchos::rcp(new Epetra_MultiVector(View,*emap,const_cast<ST*>(vals.getRawPtr()),leadingDim,mv.domain()->dim()));

    // Attach the input mv to the newly created one, to prolong its life
    // Note: the input ref must outlive the output of this routine
    Teuchos::set_extra_data(Teuchos::rcpFromRef(mv), "original multivector", inoutArg(e_mv) );
  }
  return e_mv;
}

Teuchos::RCP<Epetra_Operator>
getEpetraOperator (Thyra_LinearOp& lop,
                   const bool throw_if_not_epetra)
{
  Thyra::EpetraLinearOp* eop = dynamic_cast<Thyra::EpetraLinearOp*>(&lop);
  if (eop==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                               "Error! Could not cast input Thyra_LinearOp to Thyra::EpetraLinearOp.\n");

    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(eop->epetra_op().is_null(), std::runtime_error,
                               "Error! The Thyra::EpetraLinearOp object stores a null pointer.\n") 
    return eop->epetra_op();
  }
}

Teuchos::RCP<const Epetra_Operator>
getConstEpetraOperator (const Thyra_LinearOp& lop,
                        const bool throw_if_not_epetra)
{
  const Thyra::EpetraLinearOp* eop = dynamic_cast<const Thyra::EpetraLinearOp*>(&lop);
  if (eop==nullptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(throw_if_not_epetra, BadThyraEpetraCast,
                               "Error! Could not cast input Thyra_LinearOp to Thyra::EpetraLinearOp.\n");

    return Teuchos::null;
  } else {
    // We allow bad cast, but once cast goes through, we *expect* pointers to be valid
    TEUCHOS_TEST_FOR_EXCEPTION(eop->epetra_op().is_null(), std::runtime_error,
                               "Error! The Thyra::EpetraLinearOp object stores a null pointer.\n") 
    return eop->epetra_op();
  }
}

Teuchos::RCP<Epetra_CrsMatrix>
getEpetraMatrix (Thyra_LinearOp& lop,
                 const bool throw_if_not_epetra)
{
  auto eop = getEpetraOperator(lop,throw_if_not_epetra);
  if (!eop.is_null()) {
    auto emat = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(eop);

    // We allow bad cast, but once cast goes through, we *expect* the operator to store a crs matrix
    TEUCHOS_TEST_FOR_EXCEPTION(emat.is_null(), std::runtime_error,
                               "Error! The Thyra_EpetraLinearOp object does not store a Epetra_CrsMatrix.\n") 
    return emat;
  }
  return Teuchos::null;
}

Teuchos::RCP<const Epetra_CrsMatrix>
getConstEpetraMatrix (const Thyra_LinearOp& lop,
                      const bool throw_if_not_epetra)
{
  auto eop = getConstEpetraOperator(lop,throw_if_not_epetra);
  if (!eop.is_null()) {
    auto emat = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(eop);

    // We allow bad cast, but once cast goes through, we *expect* the operator to store a crs matrix
    TEUCHOS_TEST_FOR_EXCEPTION(emat.is_null(), std::runtime_error,
                               "Error! The Thyra_EpetraLinearOp object does not store a Epetra_CrsMatrix.\n") 
    return emat;
  }
  return Teuchos::null;
}

} // namespace Albany
