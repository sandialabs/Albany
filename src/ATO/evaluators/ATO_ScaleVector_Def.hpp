//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
ScaleVector<EvalT, Traits>::
ScaleVector(const Teuchos::ParameterList& p) :
  inVector         (p.get<std::string>                   ("Input Vector Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  outVector        (p.get<std::string>                   ("Output Vector Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::Device::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(inVector);
  this->addEvaluatedField(outVector);

  if(p.isSublist("Homogenized Constants")){
    homogenizedConstantsName = p.sublist("Homogenized Constants").get<std::string>("Name");
    useHomogenizedConstants = true;
    Albany::StateManager* stateMgr = p.get<Albany::StateManager*>("State Manager");
    Teuchos::RCP<Albany::Layouts> dl = p.get<Teuchos::RCP<Albany::Layouts> >("Data Layout");
    for(int i=1; i<=numDims; i++){
      for(int j=i; j<=numDims; j++){
        std::stringstream valname;
        valname << homogenizedConstantsName << " " << i << j;
        stateMgr->registerStateVariable(valname.str(),dl->workset_scalar, dl->dummy, "all", "scalar",0.0, false, false);
      }
    }
  } else {
    useHomogenizedConstants = false;
    coefficient = p.get<double>("Coefficient");
  }

  this->setName("ScaleVector"+PHX::typeAsString<EvalT>());

  if(p.isType<int>("Cell Forcing Column")){
    addCellForcing = true;
    cellForcingColumn = p.get<int>("Cell Forcing Column");

    TEUCHOS_TEST_FOR_EXCEPTION( 
      cellForcingColumn < 0 || cellForcingColumn >= numDims, std::logic_error,
      "Add Cell Problem Forcing: invalid column index")

    subTensor.resize(numDims);
    subTensor(cellForcingColumn) = coefficient;

  } else 
    addCellForcing = false;

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScaleVector<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(outVector,fm);
  this->utils.setFieldData(inVector,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScaleVector<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if(useHomogenizedConstants){

    std::string K(homogenizedConstantsName);

    Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> Kval(numDims,numDims);
    for(int i=0; i<numDims; i++)
      for(int j=i; j<numDims; j++){
        if( j>=i){
          std::stringstream name;
          name << K << " " << i+1 << j+1;
          Albany::MDArray coeff = (*workset.stateArrayPtr)[name.str()];
          Kval(i,j) = coeff(0);
        } else {
          Kval(j,i) = Kval(i,j);
        }
      }

    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        for (std::size_t i=0; i < numDims; ++i){
          outVector(cell,qp,i) = 0.0;
          for (std::size_t j=0; j < numDims; ++j)
            outVector(cell,qp,i) += Kval(i,j)* inVector(cell,qp,j);
        }

  } else {

    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        for (std::size_t i=0; i < numDims; ++i)
          outVector(cell,qp,i) = coefficient* inVector(cell,qp,i);
  }

  if(addCellForcing){
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        for (std::size_t i=0; i < numDims; ++i)
          outVector(cell,qp,i) -= subTensor(i);
  }
}

//**********************************************************************
}

