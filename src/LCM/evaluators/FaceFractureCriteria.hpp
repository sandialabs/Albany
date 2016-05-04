//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef FACE_FRACTURE_CRITERIA_HPP
#define FACE_FRACTURE_CRITERIA_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
  /// \brief Face-centric Fracture Criteria Evaluator
  ///
  /// Computes a fracture criterion on the faces of an element using nodal data
  ///
  template<typename EvalT, typename Traits>
  class FaceFractureCriteria : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>  {


  private:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

  public:

    FaceFractureCriteria(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

    ///
    /// Simple fracture criterion for testing
    ///
    void testFracture();

    ///
    /// Traction based criterion
    ///
    void tractionCriterion();

  private:
    unsigned int numDims;
    unsigned int numFaces;
    unsigned int numComp; // length of the vector
    unsigned int worksetSize;

    std::string criterion; // The criterion to be used

    // Input:
    PHX::MDField<ScalarT,Cell,Vertex,Dim> coord;
    PHX::MDField<ScalarT,Cell,Side,VecDim> faceAve;
    RealType yieldStrength;
    RealType fractureLimit;  // Fracture face if traction > this value

    Teuchos::RCP<shards::CellTopology> cellType;

    //Output:
    // As we can't define a boolean field on a face, define as a scalar
    PHX::MDField<ScalarT,Cell,Side> criteriaMet;

    // This is in here to trick the code to run the evaluator - does absolutely nothing
    PHX::MDField<ScalarT,Cell,QuadPoint> temp;

    // Face topology object
    const struct CellTopologyData_Subcell * sides;


  };

} // namespace LCM

#endif /* FACE_FRACTURE_CRITERIA_HPP */
