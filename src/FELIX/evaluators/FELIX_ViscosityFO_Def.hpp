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

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

const double pi = 3.1415926535897932385;
 
//**********************************************************************
template<typename EvalT, typename Traits>
ViscosityFO<EvalT, Traits>::
ViscosityFO(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  Ugrad (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  mu    (p.get<std::string> ("FELIX Viscosity QP Variable Name"), dl->qp_scalar), 
  temperature(p.get<std::string> ("Temperature Name"), dl->cell_scalar2),
  flowFactorA(p.get<std::string> ("Flow Factor Name"), dl->cell_scalar2),
  homotopyParam (1.0), 
  A(1.0), 
  n(3.0),
  flowRate_type(UNIFORM)
{
  Teuchos::ParameterList* visc_list = 
   p.get<Teuchos::ParameterList*>("Parameter List");

  std::string viscType = visc_list->get("Type", "Constant");
  std::string flowRateType = visc_list->get("Flow Rate Type", "Uniform");
  homotopyParam = visc_list->get("Glen's Law Homotopy Parameter", 1.0);
  A = visc_list->get("Glen's Law A", 1.0); 
  n = visc_list->get("Glen's Law n", 3.0);  

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (viscType == "Constant"){ 
#ifdef OUTPUT_TO_SCREEN
    *out << "Constant viscosity!" << std::endl;
#endif
    visc_type = CONSTANT;
  }
  else if (viscType == "ExpTrig") {
#ifdef OUTPUT_TO_SCREEN
   *out << "Exp trig viscosity!" << std::endl;
#endif 
    visc_type = EXPTRIG; 
  }
  else if (viscType == "Glen's Law"){
    visc_type = GLENSLAW; 
#ifdef OUTPUT_TO_SCREEN
    *out << "Glen's law viscosity!" << std::endl;
#endif
    if (flowRateType == "Uniform")
    {
    	flowRate_type = UNIFORM;
#ifdef OUTPUT_TO_SCREEN
    	*out << "Uniform Flow Rate A: " << A << std::endl;
#endif
    }
    else if (flowRateType == "From File")
    {
    	flowRate_type = FROMFILE;
    	this->addDependentField(flowFactorA);
#ifdef OUTPUT_TO_SCREEN
    	*out << "Flow Rate read in from file (exodus or ascii)." << std::endl;
#endif
    }
    else if (flowRateType == "From CISM")
    {
    	flowRate_type = FROMCISM;
    	this->addDependentField(flowFactorA);
#ifdef OUTPUT_TO_SCREEN
    	*out << "Flow Rate passed in from CISM." << std::endl;
#endif
    }
    else if (flowRateType == "Temperature Based")
    {
    	flowRate_type = TEMPERATUREBASED;
    	this->addDependentField(temperature);
#ifdef OUTPUT_TO_SCREEN
    	*out << "Flow Rate computed using Temperature field." << std::endl;
#endif
    }
#ifdef OUTPUT_TO_SCREEN
    *out << "n: " << n << std::endl; 
#endif 
  }
  coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);

  this->addDependentField(Ugrad);
  this->addDependentField(coordVec);
  this->addEvaluatedField(mu);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >("Parameter Library"); 
  
  new Sacado::ParameterRegistration<EvalT, SPL_Traits>("Glen's Law Homotopy Parameter", this, paramLib);   

  this->setName("ViscosityFO"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ViscosityFO<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(mu,fm); 
  this->utils.setFieldData(coordVec,fm); 
  if (flowRate_type == TEMPERATUREBASED)
	  this->utils.setFieldData(temperature,fm);
  if (flowRate_type == FROMFILE || flowRate_type == FROMCISM)
	  this->utils.setFieldData(flowFactorA,fm);
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename ViscosityFO<EvalT,Traits>::ScalarT& 
ViscosityFO<EvalT,Traits>::getValue(const std::string &n)
{
  return homotopyParam;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ViscosityFO<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double a = 1.0;  
  switch (visc_type) {
    case CONSTANT: 
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) 
          mu(cell,qp) = 1.0; 
      }
      break; 
    case EXPTRIG:  
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          MeshScalarT x = coordVec(cell,qp,0);
          MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
          MeshScalarT muargt = (a*a + 4.0*pi*pi - 2.0*pi*a)*sin(y2pi)*sin(y2pi) + 1.0/4.0*(2.0*pi+a)*(2.0*pi+a)*cos(y2pi)*cos(y2pi); 
          muargt = sqrt(muargt)*exp(a*x);  
          mu(cell,qp) = 1.0/2.0*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0); 
        }
      }
      break; 
    case GLENSLAW: 
      std::vector<ScalarT> flowFactorVec; //create vector of the flow factor A at each cell 
      flowFactorVec.resize(workset.numCells);
      switch (flowRate_type) {
        case UNIFORM: 
          for (std::size_t cell=0; cell < workset.numCells; ++cell) 
            flowFactorVec[cell] = 1.0/2.0*pow(A, -1.0/n);
          break; 
        case TEMPERATUREBASED:
          for (std::size_t cell=0; cell < workset.numCells; ++cell) 
	    flowFactorVec[cell] = 1.0/2.0*pow(flowRate(temperature(cell)), -1.0/n);
          break;
        case FROMFILE:
        case FROMCISM: 
          for (std::size_t cell=0; cell < workset.numCells; ++cell)  
	    flowFactorVec[cell] = 1.0/2.0*pow(flowFactorA(cell), -1.0/n);
          break;
      }
      double power = 0.5*(1.0/n - 1.0);
      if (homotopyParam == 0.0) { //set constant viscosity
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            mu(cell,qp) = flowFactorVec[cell]; 
          }
        }
      }
      else { //set Glen's law viscosity with regularization specified by homotopyParam
        ScalarT ff = pow(10.0, -10.0*homotopyParam);
        ScalarT epsilonEqpSq = 0.0; //used to define the viscosity in non-linear Stokes 
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            ScalarT& u00 = Ugrad(cell,qp,0,0); //epsilon_xx
            ScalarT& u11 = Ugrad(cell,qp,1,1); //epsilon_yy
            epsilonEqpSq = u00*u00 + u11*u11 + u00*u11; //epsilon_xx^2 + epsilon_yy^2 + epsilon_xx*epsilon_yy
            epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0)); //+0.25*epsilon_xy^2
            for (int dim = 2; dim < numDims; ++dim) //3D case
               epsilonEqpSq += 0.25*(Ugrad(cell,qp,0,dim)*Ugrad(cell,qp,0,dim) + Ugrad(cell,qp,1,dim)*Ugrad(cell,qp,1,dim) ); // + 0.25*epsilon_xz^2 + 0.25*epsilon_yz^2
            epsilonEqpSq += ff; //add regularization "fudge factor" 
            mu(cell,qp) = flowFactorVec[cell]*pow(epsilonEqpSq,  power); //non-linear viscosity, given by Glen's law  
           }
         }
      }
      break;
}
}
}
