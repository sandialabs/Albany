//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"


namespace Aeras {

const double pi = 3.1415926535897932385;
 
//**********************************************************************
template<typename EvalT, typename Traits>
ShallowWaterSource<EvalT, Traits>::
ShallowWaterSource(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  sphere_coord (p.get<std::string> ("Spherical Coord Name"), dl->qp_gradient),
  source    (p.get<std::string> ("Shallow Water Source QP Variable Name"), dl->qp_scalar)
{
  Teuchos::ParameterList* shallowWaterList =
   p.get<Teuchos::ParameterList*>("Parameter List");

   Omega = shallowWaterList->get<double>("Omega");


   lengthScale = shallowWaterList->get<double>("LengthScale");
   speedScale = shallowWaterList->get<double>("SpeedScale");

   std::string sourceTypeString = shallowWaterList->get<std::string>("SourceType", "None");
   lengthScale = shallowWaterList->get<double>("LengthScale");

   Omega = Omega*lengthScale/speedScale;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  const bool invalidString = (sourceTypeString != "None" && sourceTypeString != "TC2");
  TEUCHOS_TEST_FOR_EXCEPTION( invalidString,
                                  std::logic_error,
                                  "Unknown shallow water source string of " << sourceTypeString
                                          << " encountered. " << std::endl);
  if (sourceTypeString == "None"){
    sourceType = NONE;
  }
  else if (sourceTypeString == "TC2") {
   *out << "Using coriolis parameter from tc2." << std::endl;
   sourceType = TC2;
  }

  this->addDependentField(sphere_coord);
  this->addEvaluatedField(source);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->setName("ShallowWaterSource"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(source,fm);
  this->utils.setFieldData(sphere_coord,fm);
}

//**********************************************************************
//A concrete (non-virtual) implementation of getValue is needed for code to compile. 
//Do we really need it though for this problem...?
template<typename EvalT,typename Traits>
typename ShallowWaterSource<EvalT,Traits>::ScalarT& 
ShallowWaterSource<EvalT,Traits>::getValue(const std::string &n)
{
  static ScalarT junk(0);
  return junk;
}


//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if(sourceType == TC2) {
    double alpha = 0;  //HACK must match ic
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        const MeshScalarT theta = sphere_coord(cell, qp, 0);
        const MeshScalarT lambda = sphere_coord(cell, qp, 1);
        source(cell, qp ) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha) - sin(theta));
      }
    }
  }else if (sourceType == NONE) {
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        source(cell, qp ) = 0;
      }
    }
  }
}
}
