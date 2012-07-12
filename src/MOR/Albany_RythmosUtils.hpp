/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

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
