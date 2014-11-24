//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HYDFRACTIONRESID_HPP
#define HYDFRACTIONRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

#include "Albany_Layouts.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class HydFractionResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  HydFractionResid(Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> Temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint> Tdot;
  PHX::MDField<ScalarT,Cell,QuadPoint> Fhdot;
  PHX::MDField<ScalarT,Cell,QuadPoint> Fh;
  PHX::MDField<ScalarT,Cell,QuadPoint> JThermCond;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> TGrad;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> FhResidual;

  unsigned int numQPs, numDims, numNodes, worksetSize;
  Intrepid::FieldContainer<ScalarT> JGrad;
  Intrepid::FieldContainer<ScalarT> fh_coef;
  Intrepid::FieldContainer<ScalarT> fh_time_term;
  Intrepid::FieldContainer<ScalarT> CHZr_coef;
  Intrepid::FieldContainer<ScalarT> CH_time_term;

 //! Conductivity type
  std::string type; 

  //! Constant value
  ScalarT C_HHyd;
  ScalarT R;
  ScalarT CTSo;
  ScalarT delQ;
  ScalarT delWm;
  ScalarT stoi;
  ScalarT Vh;
  ScalarT delG;

//! Validate the name strings under "Hyd Fraction" section in xml input file, 
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidHydFractionParameters() const;

  //! Material database - holds thermal conductivity among other quantities
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

  //! storing the DataLayouts
  const Teuchos::RCP<Albany::Layouts>& dl_;

};
}

#endif
