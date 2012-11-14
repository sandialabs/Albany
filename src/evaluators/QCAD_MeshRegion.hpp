//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_MESHREGION_HPP
#define QCAD_MESHREGION_HPP

#include "Teuchos_RCP.hpp"

#include "Phalanx_MDField.hpp"
#include "Phalanx_FieldManager.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Utilities.hpp"

#include "Albany_Layouts.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace QCAD {

/** 
 * \brief A utility class that encapsulates a defined region of a mesh.  Other evaluators
 *        which operator on a mesh region use a MeshRegion instance to determine whether 
 *        a mesh point lies inside the specified region.
 */
  template<typename EvalT, typename Traits>
  class MeshRegion
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    MeshRegion(std::string coordVecName, std::string weightsName,
	       Teuchos::ParameterList& p, 
	       const Teuchos::RCP<QCAD::MaterialDatabase> matDB,
	       const Teuchos::RCP<Albany::Layouts>& dl_ );
    ~MeshRegion() { }

    void addDependentFields(PHX::EvaluatorWithBaseImpl<Traits>* evaluator);
    void postRegistrationSetup(PHX::FieldManager<Traits>& fm);

    bool elementBlockIsInRegion(std::string ebName) const;
    bool cellIsInRegion(std::size_t cell);

    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;


  private:
    std::size_t numQPs;
    std::size_t numDims;
    std::string coordVecFieldname, weightsFieldname;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
    Teuchos::RCP<Albany::Layouts> dl;

    //! Restricting to element blocks
    std::vector<std::string> ebNames; // restrict to these element blocks
    bool bQuantumEBsOnly;             // restrict to "quantum" element blocks

    //! Restricting to coordinate ranges
    bool limitX, limitY, limitZ;      // restrict along x,y,z
    double xmin, xmax, ymin, ymax, zmin, zmax;  // limits along x,y,z

    //! Restricting to a "boxed" level set of a given field
    bool bRestrictToLevelSet;
    std::string levelSetFieldname;              
    double levelSetFieldMin, levelSetFieldMax;
    PHX::MDField<ScalarT> levelSetField;    

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

    //! Evaluator utils to hide templating
    PHX::EvaluatorUtilities<EvalT,Traits> utils;
  };

}

#endif


