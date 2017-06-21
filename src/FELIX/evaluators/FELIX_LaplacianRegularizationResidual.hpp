//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_LAPLACIAN_REGULAIZATION_RESIDUAL_HPP
#define FELIX_LAPLACIAN_REGULAIZATION_RESIDUAL_HPP

//#include "FELIX_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace FELIX {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class LaplacianRegularizationResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    LaplacianRegularizationResidual(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);


  private:

 //   std::string sideName;
 //   std::vector<std::vector<int> >  sideNodes;
    
    int numCells;
    int numNodes;
    int numQPs;
    int cellDim;
 //   int sideDim;

    PHX::MDField<const ScalarT>  field;
    PHX::MDField<const ScalarT>       gradField;
    PHX::MDField<const ParamScalarT>  forcing;
    PHX::MDField<const MeshScalarT>   gradBF;
    PHX::MDField<const MeshScalarT>   w_measure;

    PHX::MDField<ScalarT> residual;

    ScalarT p_reg, reg;
    double laplacian_coeff, mass_coeff;
    int offset;
  };

} // Namespace FELIX

#endif // FELIX_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
