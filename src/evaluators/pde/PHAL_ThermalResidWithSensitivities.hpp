//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_THERMALRESIDWITHSENSITIVITIES_HPP
#define PHAL_THERMALRESIDWITHSENSITIVITIES_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ThermalResidWithSensitivities : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
                    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  ThermalResidWithSensitivities(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

  typename EvalT::ScalarT& getValue(const std::string &n);

private:

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT value;  
  
  // Input:
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint>                Tdot;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<ScalarT const, Cell, QuadPoint, Dim>           TGrad;
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim>       coordVec;
  Teuchos::Array<double> kappa;  // Thermal Conductivity array
  double                 C;      // Heat Capacity
  double                 rho;    // Density

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint> Source;
  PHX::MDField<ScalarT, Cell, Node> TResidual;

  unsigned int numQPs, numDims, numNodes, worksetSize;
  enum FTYPE {NONE, ONEDCOST, TWODCOSTEXPT};
  FTYPE force_type;
};
}

#endif
