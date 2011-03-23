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


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
PHAL::TEProp<EvalT, Traits>::
TEProp(Teuchos::ParameterList& p) :
  rhoCp(p.get<std::string>("QP Variable Name 3"),
	      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  permittivity(p.get<std::string>("QP Variable Name 2"),
	      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  thermalCond(p.get<std::string>("QP Variable Name"),
	      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  Temp(p.get<std::string>("Temperature Variable Name"),
	      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  coordVec(p.get<std::string>("Coordinate Vector Name"),
            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout"))
{
  Teuchos::ParameterList* teprop_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  mats = teprop_list->get<int>("Number of Materials");;

  Teuchos::Array<double> dbl_elecCs = Teuchos::getArrayFromStringParameter<double> (*teprop_list,
                           "Electrical Conductivity", mats, true);
  elecCs.resize(dbl_elecCs.size());
  for (int i=0; i<dbl_elecCs.size(); i++) elecCs[i] = dbl_elecCs[i];

  thermCs  = Teuchos::getArrayFromStringParameter<double> (*teprop_list,
                           "Thermal Conductivity", mats, true);
  rhoCps   = Teuchos::getArrayFromStringParameter<double> (*teprop_list,
                           "Rho Cp", mats, true);
  factor  = Teuchos::getArrayFromStringParameter<double> (*teprop_list,
                           "Coupling Factor", mats, true);
  xBounds = Teuchos::getArrayFromStringParameter<double> (*teprop_list,
                           "X Bounds", mats+1, true);

  // Add Electrical Conductivity as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  for (int i=0; i<mats; i++) {
      std::stringstream ss;
      ss << "Electrical Conductivity of Material " << i;
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss.str(), this, paramLib);
  }


  this->addDependentField(Temp);
  this->addDependentField(coordVec);
  this->addEvaluatedField(rhoCp);
  this->addEvaluatedField(permittivity);
  this->addEvaluatedField(thermalCond);
  this->setName("TEProp"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::TEProp<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(permittivity,fm);
  this->utils.setFieldData(thermalCond,fm);
  this->utils.setFieldData(rhoCp,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(Temp,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void PHAL::TEProp<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  unsigned int numCells = workset.numCells;
   
  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      int mat = whichMat(coordVec(cell,qp,0));
        permittivity(cell,qp) = elecCs[mat] / (1.0 + factor[mat] * Temp(cell,qp));
        thermalCond(cell,qp) = thermCs[mat];
        rhoCp(cell,qp) = rhoCps[mat];
      }
    }
}

// **********************************************************************
template<typename EvalT, typename Traits>
int PHAL::TEProp<EvalT, Traits>::
whichMat(const MeshScalarT& x)
{
  TEST_FOR_EXCEPTION(x<xBounds[0] || x>xBounds[mats], std::logic_error,
     "Quadrature point " << x << " not within bounds \n");
  for (int i=0; i<mats; i++) 
     if (x<xBounds[i+1]) return i;
  TEST_FOR_EXCEPTION(true, std::logic_error,
     "Quadrature point " << x << " not within bounds \n");
  return -1;
}
// **********************************************************************
template<typename EvalT,typename Traits>
typename PHAL::TEProp<EvalT,Traits>::ScalarT& 
PHAL::TEProp<EvalT,Traits>::getValue(const std::string &n)
{
  int mat;
  for (int i=0; i<mats; i++) {
      std::stringstream ss;
      ss << "Electrical Conductivity of Material " << i;
      if (n == ss.str()) { mat=i; break; }
   }
   return elecCs[mat];
}

// **********************************************************************
// **********************************************************************
