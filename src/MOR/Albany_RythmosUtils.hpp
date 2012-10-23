//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_RYTHMOSUTILS_HPP
#define ALBANY_RYTHMOSUTILS_HPP

#include "Rythmos_StepperBase.hpp"

#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_VectorBase.hpp"

#include "Thyra_EpetraThyraWrappers.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

template <typename Scalar>
Teuchos::RCP<const Thyra::VectorBase<Scalar> > getRythmosState(const Teuchos::RCP<const Thyra::VectorBase<Scalar> > &in) {
  typedef Thyra::DefaultProductVector<Scalar> DPV;
  const Teuchos::RCP<const DPV> in_with_sens = Teuchos::rcp_dynamic_cast<const DPV>(in);
  if (Teuchos::nonnull(in_with_sens)) {
    return in_with_sens->getVectorBlock(0);
  } else {
    return in;
  }
}

} // namespace Albany

#endif /* ALBANY_RYTHMOSUTILS_HPP */
