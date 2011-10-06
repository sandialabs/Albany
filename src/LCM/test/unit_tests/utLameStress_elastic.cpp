/********************************************************************\
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
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_DiscretizationFactory.hpp"

#include "LCM/evaluators/LameStress.hpp"
#include "Tensor.h"

using namespace std;

namespace {

// Create a DefGrad evaluator that does not have any dependent fields.
// This will allow for calls to the LameStress evaluator without constructing the entire evaluation tree.
// Instead, the deformation gradient will be set explicitly in DefGradUnitTest, and will then be passed
// to the LameStress evaluator via the FieldManager.
template<typename EvalT, typename Traits>
class DefGradUnitTest : public PHX::EvaluatorWithBaseImpl<Traits>,
		public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DefGradUnitTest(const Teuchos::ParameterList& p) :
    defgrad(p.get<std::string>("DefGrad Name"), p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
  {
    Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    worksetSize  = dims[0];
    numQPs  = dims[1];
    numDims = dims[2];

    this->addEvaluatedField(defgrad);

    this->setName("DefGrad"+PHX::TypeString<EvalT>::value);
  }

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(defgrad,fm);
  }

  void evaluateFields(typename Traits::EvalData workset)
  {
    // Set the deformation gradient
    for(std::size_t cell=0; cell<workset.numCells; ++cell){
      for(std::size_t qp=0; qp<numQPs; ++qp){
        defgrad(cell,qp,0,0) = 1.010050167084168;
        defgrad(cell,qp,0,1) = 0.0;
        defgrad(cell,qp,0,2) = 0.0;
        defgrad(cell,qp,1,0) = 0.0;
        defgrad(cell,qp,1,1) = 0.99750312239746;
        defgrad(cell,qp,1,2) = 0.0;
        defgrad(cell,qp,2,0) = 0.0;
        defgrad(cell,qp,2,1) = 0.0;
        defgrad(cell,qp,2,2) = 0.99750312239746;
      }
    }
  }

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int worksetSize;
};


TEUCHOS_UNIT_TEST( LameStress_elastic, Instantiation )
{
  // Set up the data layout
  const int worksetSize = 1;
  const int numQPts = 1;
  const int numDim = 3;
  Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint> > qp_scalar =
    Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint>(worksetSize, numQPts));
  Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint, Dim, Dim> > qp_tensor =
    Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim, Dim>(worksetSize, numQPts, numDim, numDim));

  // Instantiate the required evaluators with EvalT = PHAL::AlbanyTraits::Residual and Traits = PHAL::AlbanyTraits

  // DefGradUnitTest evaluator
  Teuchos::ParameterList defGradUnitTestParameterList("DefGrad");
  defGradUnitTestParameterList.set<bool>("avgJ Name", false);
  defGradUnitTestParameterList.set<bool>("volavgJ Name", false);
  defGradUnitTestParameterList.set<string>("Weights Name", "Weights");
  defGradUnitTestParameterList.set<string>("Gradient QP Variable Name", "Displacement Gradient");
  defGradUnitTestParameterList.set<string>("DefGrad Name", "Deformation Gradient");
  defGradUnitTestParameterList.set<string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
  defGradUnitTestParameterList.set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", qp_scalar);
  defGradUnitTestParameterList.set< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout", qp_tensor);
  Teuchos::RCP<DefGradUnitTest<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > defGradUnitTest = 
    Teuchos::rcp(new DefGradUnitTest<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(defGradUnitTestParameterList));

  // LameStress evaluator
  Teuchos::RCP<Teuchos::ParameterList> lameStressParameterList = Teuchos::rcp(new Teuchos::ParameterList("Stress"));
  //int type = LameStressUnitTestFactoryTraits<PHAL::AlbanyTraits>::id_lame_stress;
  //lameStressParameterList->set<int>("Type", type);
  lameStressParameterList->set<string>("DefGrad Name", "Deformation Gradient");
  lameStressParameterList->set<string>("Stress Name", "Stress");
  lameStressParameterList->set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", qp_scalar);
  lameStressParameterList->set< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout", qp_tensor);
  lameStressParameterList->set<string>("Lame Material Model", "Elastic");
  Teuchos::ParameterList& materialModelParametersList = lameStressParameterList->sublist("Lame Material Parameters");
  materialModelParametersList.set<double>("Youngs Modulus", 1.0);
  materialModelParametersList.set<double>("Poissons Ratio", 0.25);
  Teuchos::RCP<LCM::LameStress<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> > lameStress = 
    Teuchos::rcp(new LCM::LameStress<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>(*lameStressParameterList));

  // Instantiate a field manager.
  PHX::FieldManager<PHAL::AlbanyTraits> fieldManager;

  // Register the evaluators with the field manager
  fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(defGradUnitTest);
  fieldManager.registerEvaluator<PHAL::AlbanyTraits::Residual>(lameStress);

  // Set the LameStress evaluated fields as required fields
  for(std::vector<Teuchos::RCP<PHX::FieldTag> >::const_iterator it = lameStress->evaluatedFields().begin() ; it != lameStress->evaluatedFields().end() ; it++)
    fieldManager.requireField<PHAL::AlbanyTraits::Residual>(**it);
 
  PHAL::AlbanyTraits::SetupData setupData = "Test String";
  fieldManager.postRegistrationSetup(setupData);

  Albany::StateManager stateMgr;

  // Stress and DefGrad are required for all LAME models
  stateMgr.registerStateVariable("Stress", qp_tensor, "zero", true);
  stateMgr.registerStateVariable("Deformation Gradient", qp_tensor, "identity", true);

  // Add material-model specific state variables
  string lameMaterialModelName = lameStressParameterList->get<string>("Lame Material Model");
  std::vector<std::string> lameMaterialModelStateVariableNames = LameUtils::getStateVariableNames(lameMaterialModelName, materialModelParametersList);
  std::vector<double> lameMaterialModelStateVariableInitialValues = LameUtils::getStateVariableInitialValues(lameMaterialModelName, materialModelParametersList);
  for(unsigned int i=0 ; i<lameMaterialModelStateVariableNames.size() ; ++i){
    stateMgr.registerStateVariable(lameMaterialModelStateVariableNames[i],
                                   qp_scalar,
                                   Albany::doubleToInitString(lameMaterialModelStateVariableInitialValues[i]),
                                   true);
  }

  Teuchos::RCP<Teuchos::ParameterList> discretizationParameterList = Teuchos::rcp(new Teuchos::ParameterList("Discretization"));
  discretizationParameterList->set<int>("1D Elements", 1);
  discretizationParameterList->set<int>("2D Elements", 1);
  discretizationParameterList->set<int>("3D Elements", 1);
  discretizationParameterList->set<string>("Method", "STK3D");
  discretizationParameterList->set<string>("Exodus Output File Name", "unitTestOutput.exo"); // Is this required?

  Teuchos::RCP<Epetra_Comm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  Albany::DiscretizationFactory discretizationFactory(discretizationParameterList, comm);

  discretizationFactory.createMeshSpecs();

  int numberOfEquations = 3;

  Teuchos::RCP<Albany::AbstractDiscretization> discretization =
    discretizationFactory.createDiscretization(numberOfEquations,
                                               stateMgr.getStateInfoStruct());

  stateMgr.setStateArrays(discretization);

  PHAL::Workset workset;
  workset.numCells = worksetSize;
  workset.stateArrayPtr = &stateMgr.getStateArray(0);

  void* voidPtr(0);

  fieldManager.preEvaluate<PHAL::AlbanyTraits::Residual>(voidPtr);
  fieldManager.evaluateFields<PHAL::AlbanyTraits::Residual>(workset);
  fieldManager.postEvaluate<PHAL::AlbanyTraits::Residual>(voidPtr);

  PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT,Cell,QuadPoint,Dim,Dim> stressField("Stress", qp_tensor);
  fieldManager.getFieldData<PHAL::AlbanyTraits::Residual::ScalarT, PHAL::AlbanyTraits::Residual, Cell, QuadPoint, Dim, Dim>(stressField);

  typedef PHX::MDField<PHAL::AlbanyTraits::Residual::ScalarT>::size_type size_type;
  std::vector<size_type> stressFieldDimensions;
  stressField.dimensions(stressFieldDimensions);

  LCM::Tensor<PHAL::AlbanyTraits::Residual::ScalarT>
    expectedStress(materialModelParametersList.get<double>("Youngs Modulus") * 0.01,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0);

  for(size_type cell=0; cell<worksetSize; ++cell){
    for(size_type qp=0; qp<numQPts; ++qp){

      std::cout << "Stress tensor at cell " << cell << ", quadrature point " << qp << ":" << endl;
      std::cout << "  " << stressField(cell, qp, 0, 0);
      std::cout << "  " << stressField(cell, qp, 0, 1);
      std::cout << "  " << stressField(cell, qp, 0, 2) << endl;
      std::cout << "  " << stressField(cell, qp, 1, 0);
      std::cout << "  " << stressField(cell, qp, 1, 1);
      std::cout << "  " << stressField(cell, qp, 1, 2) << endl;
      std::cout << "  " << stressField(cell, qp, 2, 0);
      std::cout << "  " << stressField(cell, qp, 2, 1);
      std::cout << "  " << stressField(cell, qp, 2, 2) << endl;

      std::cout << "Expected result:" << endl;
      std::cout << "  " << expectedStress(0, 0);
      std::cout << "  " << expectedStress(0, 1);
      std::cout << "  " << expectedStress(0, 2) << endl;
      std::cout << "  " << expectedStress(1, 0);
      std::cout << "  " << expectedStress(1, 1);
      std::cout << "  " << expectedStress(1, 2) << endl;
      std::cout << "  " << expectedStress(2, 0);
      std::cout << "  " << expectedStress(2, 1);
      std::cout << "  " << expectedStress(2, 2) << endl;

      std::cout << endl;

      double tolerance = 1.0e-15;
      for(size_type i=0 ; i<numDim ; ++i){
        for(size_type j=0 ; j<numDim ; ++j){
          TEST_COMPARE( fabs(stressField(cell, qp, i, j) - expectedStress(i, j)), <=, tolerance );
        }
      }

    }
  }
  std::cout << endl;

}

} // namespace
