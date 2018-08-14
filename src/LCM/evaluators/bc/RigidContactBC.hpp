//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef RIGIDCONTACTBC_HPP
#define RIGIDCONTACTBC_HPP

#include "PHAL_Neumann.hpp"

#include "Teuchos_TwoDArray.hpp"

namespace LCM {

/** \brief contact with a smooth rigid object boundary condition evaluator

*/

template <typename EvalT, typename Traits>
class RigidContactBC_Base : public PHAL::Neumann<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  RigidContactBC_Base(Teuchos::ParameterList& p);

  void
  computeVal(RealType time);
  void
  computeCoordVal(RealType time);

 protected:
  std::vector<RealType>        timeValues;
  Teuchos::TwoDArray<RealType> BCValues;
};

template <typename EvalT, typename Traits>
class RigidContactBC : public RigidContactBC_Base<EvalT, Traits>
{
 public:
  RigidContactBC(Teuchos::ParameterList& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT ScalarT;
};

}  // namespace LCM

#endif
