//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

#include "QCAD_MathVector.hpp"
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

    //! Restricting to xy-polygon (still need zmin & zmax)
    bool bRestrictToXYPolygon;
    std::vector<mathVector> xyPolygon; //polygon of (x,y) points 

    //! Restricting to a "boxed" level set of a given field
    bool bRestrictToLevelSet;
    std::string levelSetFieldname;              
    double levelSetFieldMin, levelSetFieldMax;
    PHX::MDField<ScalarT> levelSetField;    

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

    //! Evaluator utils to hide templating
    PHX::EvaluatorUtilities<EvalT,Traits> utils;

  public:
    static Teuchos::RCP<const Teuchos::ParameterList> getValidParameters()
    {
      const int MAX_POLYGON_PTS = 20;
      Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid MeshRegion Params"));;

      validPL->set<std::string>("Operation Domain", "", "Deprecated - does nothing"); //TODO: remove?
      
      validPL->set<std::string>("Element Block Name", "", "Element block name to restrict region to");
      validPL->set<std::string>("Element Block Names", "", "Element block names to restrict region to");
      validPL->set<bool>("Quantum Element Blocks Only", false, "Restricts region to quantum element blocks");
      
      validPL->set<double>("x min", 0.0, "Box domain minimum x coordinate");
      validPL->set<double>("x max", 0.0, "Box domain maximum x coordinate");
      validPL->set<double>("y min", 0.0, "Box domain minimum y coordinate");
      validPL->set<double>("y max", 0.0, "Box domain maximum y coordinate");
      validPL->set<double>("z min", 0.0, "Box domain minimum z coordinate");
      validPL->set<double>("z max", 0.0, "Box domain maximum z coordinate");

      Teuchos::ParameterList& polyPL = validPL->sublist("XY Polygon");
      polyPL.set<int>("Number of Points", 0, "The number of points in the polygon");
      for(int i=0; i<MAX_POLYGON_PTS; ++i) {
	polyPL.set< Teuchos::Array<double> >(Albany::strint("Point",i), Teuchos::Array<double>(2,0), "(x,y) point of polygon");
      }
      
      validPL->set<std::string>("Level Set Field Name", "<field name>","Scalar Field to use for level set region");
      validPL->set<double>("Level Set Field Minimum", 0.0, "Minimum value of field to include in region");
      validPL->set<double>("Level Set Field Maximum", 0.0, "Maximum value of field to include in region");
      
      return validPL;
    }

  };

}

#endif


