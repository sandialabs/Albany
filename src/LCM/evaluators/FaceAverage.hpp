//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef FACE_AVERAGE_HPP_
#define FACE_AVERAGE_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace LCM {
/** \brief Computes the face average of a nodal value
 *
 * \param[in] nodal variable
 * \param[out] Face averaged variable
 */

template<typename EvalT, typename Traits>
class FaceAverage : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

      FaceAverage(const Teuchos::ParameterList& p);

      void postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& vm);

      void evaluateFields(typename Traits::EvalData d);

    private:

      typedef typename EvalT::ScalarT ScalarT;
      typedef typename EvalT::MeshScalarT MeshScalarT;

      unsigned int numNodes;
      unsigned int numDims;
      unsigned int numFaces;
      unsigned int numComp; // length of the vector
      unsigned int worksetSize;
      unsigned int faceDim;
      unsigned int numFaceNodes;
      unsigned int numQPs;

      // Input:
      // Coordinates in the reference configuration
      PHX::MDField<ScalarT,Cell,Vertex,Dim> coordinates;

      // The field that was projected to the nodes
      PHX::MDField<ScalarT,Cell,Node,VecDim> projected;
      //Numerical integration rule
      Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> >> cubature;
      // FE basis
      Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>> intrepidBasis;
      // The cell type
      Teuchos::RCP<shards::CellTopology> cellType;

      //Output:
      // As a test, output the face average of the nodal field
      PHX::MDField<ScalarT,Cell,Face,VecDim> faceAve;

      // This is in here to trick the code to run the evaluator - does absolutely nothing
      PHX::MDField<ScalarT,Cell,QuadPoint> temp;

      // For creating the quadrature weights
      Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refPoints;
      Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refWeights;
      Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refValues;

      // Face topology data
      const struct CellTopologyData_Subcell * sides;

};

} // namespace LCM


#endif /* FACEAVERAGE_HPP_ */
