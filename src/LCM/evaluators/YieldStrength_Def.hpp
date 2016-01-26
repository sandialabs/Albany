//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"
#include <typeinfo>

namespace LCM {

template<typename EvalT, typename Traits>
YieldStrength<EvalT, Traits>::
YieldStrength(Teuchos::ParameterList& p) :
  yieldStrength(p.get<std::string>("QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Yield Strength Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0);

    // Add Yield Strength as a Sacado-ized parameter
    this->registerSacadoParameter("Yield Strength", paramLib);
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*elmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Yield Strength KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Invalid yield strength type " << type);
  } 

  // Optional dependence on Temperature (Y = Y + dYdT * T)
  // Switched ON by sending Temperature field in p

  if ( p.isType<std::string>("QP Temperature Name") ) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      tmp(p.get<std::string>("QP Temperature Name"), scalar_dl);
    Temperature = tmp;
    this->addDependentField(Temperature);
    isThermoElastic = true;
    dYdT_value = elmd_list->get("dYdT Value", 0.0);
    refTemp = p.get<RealType>("Reference Temperature", 0.0);
    this->registerSacadoParameter("dYdT Value", paramLib);
  }
  else {
    isThermoElastic=false;
    dYdT_value=0.0;
  }

  if ( p.isType<std::string>("Lattice Concentration Name") ) {
      Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
      PHX::MDField<ScalarT,Cell,QuadPoint>
        tmp(p.get<std::string>("Lattice Concentration Name"), scalar_dl);
      CL = tmp;
      this->addDependentField(CL);
      CLname = p.get<std::string>("Lattice Concentration Name")+"_old";
  //    PHX::MDField<ScalarT,Cell,QuadPoint>
  //      tmp(p.get<std::string>("Trapped Concentration Name_old"), scalar_dl);
  //    CT = tmp;
  //    this->addDependentField(CT);
  //    CTname = p.get<std::string>("Trapped Concentration Name")+"_old";


      isDiffuseDeformation = true;
      zeta = elmd_list->get("zeta Value", 1.0);
      this->registerSacadoParameter("zeta Value", paramLib);
    }
    else {
     isDiffuseDeformation=false;
      zeta=1.0;
    }


  this->addEvaluatedField(yieldStrength);
  this->setName("Yield Strength"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void YieldStrength<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(yieldStrength,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature,fm);
  if (isDiffuseDeformation) this->utils.setFieldData(CL,fm);
//  if (isDiffuseDeformation) this->utils.setFieldData(CT,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void YieldStrength<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  bool print = false;
  //if (typeid(ScalarT) == typeid(RealType)) print = true;

  if (print)
    std::cout << " *** YieldStrength *** " << std::endl;

  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	yieldStrength(cell,qp) = constant_value;
      }
    }
  }
#ifdef ALBANY_STOKHOS
  else {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (int i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
	yieldStrength(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isThermoElastic) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	yieldStrength(cell,qp) -= dYdT_value * (Temperature(cell,qp) - refTemp);

        if (print)
        {
          std::cout << "    Y   : " << yieldStrength(cell,qp) << std::endl;
          std::cout << "    temp: " << Temperature(cell,qp) << std::endl;
          std::cout << "    dYdT: " << dYdT_value << std::endl;
          std::cout << "    refT: " << refTemp << std::endl;
        }
      }
    }
  }
  if (isDiffuseDeformation) {

	  Albany::MDArray CLold   = (*workset.stateArrayPtr)[CLname];

      for (int cell=0; cell < numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
 //       	yieldStrength(cell,qp) = constant_value*( 1.0 + (zeta-1.0)*CL(cell,qp)   );
        	yieldStrength(cell,qp) -= constant_value*(zeta-1.0)*(CL(cell,qp) -CLold(cell,qp)  );

          if (print)
          {
            std::cout << "    Y   : " << yieldStrength(cell,qp) << std::endl;
            std::cout << "    CT  : " << CT(cell,qp) << std::endl;
            std::cout << "   zeta : " << zeta << std::endl;
          }
        }
      }
    }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename YieldStrength<EvalT,Traits>::ScalarT& 
YieldStrength<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Yield Strength")
    return constant_value;
  else if (n == "dYdT Value")
    return dYdT_value;
  else if (n == "zeta Value")
      return zeta;
#ifdef ALBANY_STOKHOS
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Yield Strength KL Random Variable",i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in YieldStrength::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

