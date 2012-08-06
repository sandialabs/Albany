/*
 * FaceFractureCriteria.hpp
 *
 *  Created on: Jul 20, 2012
 *      Author: jrthune
 */

#ifndef FACE_FRACTURE_CRITERIA_HPP
#define FACE_FRACTURE_CRITERIA_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace LCM {
/** \brief Face-centric Fracture Criteria Evaluator
 *
 *   Computes a fracture criterion on the faces of an element using nodal data
 *
 */


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

    /** Simple fracture criterion for testing
     * \param[in] faceAve - the face averaged projected variable
     * \param[out] criteriaMet - boolean value denoting criterion state
     *
     */
    void testFracture();

    /** Traction based criterion
     * \param[in] faceAve - the face averaged projected variable
     * \param[in] coord - the nodal coordinates
     * \param[out] criteriaMet - boolean value denoting criterion state
     */
    void tractionCriterion();

  private:
    unsigned int numDims;
    unsigned int numFaces;
    unsigned int numComp; // length of the vector
    unsigned int worksetSize;

    std::string criterion; // The criterion to be used

    // Input:
    PHX::MDField<ScalarT,Cell,Vertex,Dim> coord;
    PHX::MDField<ScalarT,Cell,Face,VecDim> faceAve;
    RealType yieldStrength;
    RealType fractureLimit;  // Fracture face if traction > this value

   Teuchos::RCP<shards::CellTopology> cellType;

    //Output:
    // As we can't define a boolean field on a face, define as a scalar
    PHX::MDField<ScalarT,Cell,Face> criteriaMet;

    // This is in here to trick the code to run the evaluator - does absolutely nothing
    PHX::MDField<ScalarT,Cell> temp;

    // Face topology object
    const struct CellTopologyData_Subcell * sides;


};

} // namespace LCM

#endif /* FACE_FRACTURE_CRITERIA_HPP */
