/********************************************************************  \
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

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Epetra_MpiComm.h>
#include <Phalanx.hpp>
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "LCM/evaluators/Localization.hpp"
#include "LCM/evaluators/SetField.hpp"
#include "Tensor.h"


using namespace std;

namespace {

  TEUCHOS_UNIT_TEST( Localization, Instantiation )
  {
    // Set up the data layout
    const int worksetSize = 1;
    const int numQPts = 4;
    const int numDim = 3;
    const int numVertices = 8;
    Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint> > qp_scalar =
      Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint>(worksetSize, numQPts));
    Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint, Dim, Dim> > qp_tensor =
      Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim, Dim>(worksetSize, numQPts, numDim, numDim));
    Teuchos::RCP<PHX::MDALayout<Cell, Vertex, Dim> > vertices_vector =
      Teuchos::rcp(new PHX::MDALayout<Cell, Vertex, Dim>(worksetSize, numVertices, numDim));

    // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits
    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> referenceCoords(24);
    referenceCoords[0] = -0.5;
    referenceCoords[1] = 0.0;
    referenceCoords[2] = -0.5;
    referenceCoords[3] = -0.5;
    referenceCoords[4] = 0.0;
    referenceCoords[5] = 0.5;
    referenceCoords[6] = 0.5;
    referenceCoords[7] = 0.0;
    referenceCoords[8] = 0.5;
    referenceCoords[9] = 0.5;
    referenceCoords[10] = 0.0;
    referenceCoords[11] = -0.5;
    referenceCoords[12] = -0.5;
    referenceCoords[13] = 0.0;
    referenceCoords[14] = -0.5;
    referenceCoords[15] = -0.5;
    referenceCoords[16] = 0.0;
    referenceCoords[17] = 0.5;
    referenceCoords[18] = 0.5;
    referenceCoords[19] = 0.0;
    referenceCoords[20] = 0.5;
    referenceCoords[21] = 0.5;
    referenceCoords[22] = 0.0;
    referenceCoords[23] = -0.5;

    Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> currentCoords(24);
    const double eps = 0.1;
    referenceCoords[0] = -0.5;
    currentCoords[1] = eps;
    currentCoords[2] = 0.5;
    currentCoords[3] = 0.5;
    currentCoords[4] = eps;
    currentCoords[5] = 0.5;
    currentCoords[6] = 0.5;
    currentCoords[7] = eps;
    currentCoords[8] = 0.5;
    currentCoords[9] = 0.5;
    currentCoords[10] = eps;
    currentCoords[11] = 0.5;
    currentCoords[12] = 0.5;
    currentCoords[13] = eps;
    currentCoords[14] = 0.5;
    currentCoords[15] = 0.5;
    currentCoords[16] = eps;
    currentCoords[17] = 0.5;
    currentCoords[18] = 0.5;
    currentCoords[19] = eps;
    currentCoords[20] = 0.5;
    currentCoords[21] = 0.5;
    currentCoords[22] = eps;
    currentCoords[23] = 0.5;


    // SetField evaluator, which will be used to manually assign a value to the Ref Coord field
    Teuchos::ParameterList setRefFieldParameterList("SetFieldRefCoords");
    setRefFieldParameterList.set<string>("Evaluated Field Name", "Reference Coords");
    setRefFieldParameterList.set< Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", vertices_vector);
    setRefFieldParameterList.set< Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", referenceCoords);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldRefCoords = 
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setRefFieldParameterList));

    // SetField evaluator, which will be used to manually assign a value to the Cur Coord field
    Teuchos::ParameterList setCurFieldParameterList("SetFieldCurCoords");
    setCurFieldParameterList.set<string>("Evaluated Field Name", "Current Coords");
    setCurFieldParameterList.set< Teuchos::RCP<PHX::DataLayout> >("Evaluated Field Data Layout", vertices_vector);
    setCurFieldParameterList.set< Teuchos::ArrayRCP<PHAL::AlbanyTraits::Residual::ScalarT> >("Field Values", currentCoords);
    Teuchos::RCP<LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > setFieldCurCoords = 
      Teuchos::rcp(new LCM::SetField<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(setCurFieldParameterList));

    // stuff for the localization evaluator
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
    intrepidBasis = Teuchos::rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, Intrepid::FieldContainer<RealType> >() );
    Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology( shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    Intrepid::DefaultCubatureFactory<RealType> cubFactory;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, 3);

    // Localization evaluator
    Teuchos::RCP<Teuchos::ParameterList> localizationParameterList = Teuchos::rcp(new Teuchos::ParameterList("Localization"));
    localizationParameterList->set<string>("Current Coordinates Name", "Current Coords");
    localizationParameterList->set<string>("Reference Coordinates Name", "Reference Coords");
    localizationParameterList->set<string>("Deformation Gradient Name", "Deformation Gradient");
    localizationParameterList->set< Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    localizationParameterList->set< Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >("Intrepid Basis",intrepidBasis);
    localizationParameterList->set< Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    localizationParameterList->set< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout", qp_tensor);
    localizationParameterList->set< Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout", vertices_vector);
    Teuchos::RCP<LCM::Localization<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > localization = 
      Teuchos::rcp(new LCM::Localization<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(*localizationParameterList));

    // Instantiate a field manager.
    PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

    // Register the evaluators with the field manager
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldRefCoords);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(setFieldCurCoords);
    fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(localization);

    // Set the Localization evaluated fields as required fields
    for(std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = localization->evaluatedFields().begin() ; it != localization->evaluatedFields().end() ; it++)
      fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);
 
    // Call postRegistrationSetup on the evaluators
    // JTO - I don't know what "Test String" is meant for...
    PHAL::AlbanyTraits::SetupData setupData = "Test String";
    fieldManager.postRegistrationSetup(setupData);

    // Create a workset
    PHAL::Workset workset;
    workset.numCells = worksetSize;

    // Call the evaluators, evaluateFields() is the function that computes things
    fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
    fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(workset);

    // Pull the stress from the FieldManager
    PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,QuadPoint,Dim,Dim> defGradField("Deformation Gradient", qp_tensor);
    fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT, PHAL::AlbanyTraits::Residual, Cell, QuadPoint, Dim, Dim>(defGradField);

    // Record the expected stress, which will be used to check the computed stress
    LCM::Tensor<PHAL::AlbanyTraits::Residual::ScalarT>
      expectedDefGrad(1.0,
                      0.0,
                      0.0,
                      0.0,
                      1.0,
                      0.0,
                      0.0,
                      0.0,
                      1.0);

    // Check the computed stresses
    typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
    for(size_type cell=0; cell< worksetSize; ++cell){
      for(size_type qp=0; qp < numQPts; ++qp){

        std::cout << "Stress tensor at cell " << cell << ", quadrature point " << qp << ":" << endl;
        std::cout << "  " << defGradField(cell, qp, 0, 0);
        std::cout << "  " << defGradField(cell, qp, 0, 1);
        std::cout << "  " << defGradField(cell, qp, 0, 2) << endl;
        std::cout << "  " << defGradField(cell, qp, 1, 0);
        std::cout << "  " << defGradField(cell, qp, 1, 1);
        std::cout << "  " << defGradField(cell, qp, 1, 2) << endl;
        std::cout << "  " << defGradField(cell, qp, 2, 0);
        std::cout << "  " << defGradField(cell, qp, 2, 1);
        std::cout << "  " << defGradField(cell, qp, 2, 2) << endl;

        std::cout << "Expected result:" << endl;
        std::cout << "  " << expectedDefGrad(0, 0);
        std::cout << "  " << expectedDefGrad(0, 1);
        std::cout << "  " << expectedDefGrad(0, 2) << endl;
        std::cout << "  " << expectedDefGrad(1, 0);
        std::cout << "  " << expectedDefGrad(1, 1);
        std::cout << "  " << expectedDefGrad(1, 2) << endl;
        std::cout << "  " << expectedDefGrad(2, 0);
        std::cout << "  " << expectedDefGrad(2, 1);
        std::cout << "  " << expectedDefGrad(2, 2) << endl;

        std::cout << endl;

        double tolerance = 1.0e-15;
        for(size_type i=0 ; i<numDim ; ++i){
          for(size_type j=0 ; j<numDim ; ++j){
            TEST_COMPARE( fabs(defGradField(cell, qp, i, j) - expectedDefGrad(i, j)), <=, tolerance );
          }
        }

      }
    }
    std::cout << endl;

  }

} // namespace
