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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "LCM/utils/Tensor.h"

#include <Sacado_MathFunctions.hpp>

#include <typeinfo>

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermoMechanicalStress<EvalT, Traits>::
ThermoMechanicalStress(const Teuchos::ParameterList& p) :
  F_array          (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J_array          (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  shearModulus     (p.get<std::string>                   ("Shear Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  bulkModulus      (p.get<std::string>                   ("Bulk Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  temperature      (p.get<std::string>                   ("Temperature Name"),
		    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  yieldStrength    (p.get<std::string>                   ("Yield Strength Name"),
		    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  hardeningModulus (p.get<std::string>                   ("Hardening Modulus Name"),
		    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  deltaTime        (p.get<std::string>                   ("Delta Time Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Workset Scalar Data Layout")),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  Fp               (p.get<std::string>                   ("Fp Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  eqps             (p.get<std::string>                   ("eqps Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  mechSource       (p.get<std::string>                   ("Mechanical Source Name"),
		    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  thermalExpansionCoeff (p.get<RealType>("Thermal Expansion Coefficient") ),
  refTemperature (p.get<RealType>("Reference Temperature") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(F_array);
  this->addDependentField(J_array);
  this->addDependentField(shearModulus);
  this->addDependentField(bulkModulus);
  this->addDependentField(yieldStrength);
  this->addDependentField(hardeningModulus);
  this->addDependentField(temperature);
  this->addDependentField(deltaTime);

  fpName = p.get<std::string>("Fp Name")+"_old";
  eqpsName = p.get<std::string>("eqps Name")+"_old";
  this->addEvaluatedField(stress);
  this->addEvaluatedField(Fp);
  this->addEvaluatedField(eqps);
  this->addEvaluatedField(mechSource);

  this->setName("ThermoMechanical Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(Fp,fm);
  this->utils.setFieldData(eqps,fm);
  this->utils.setFieldData(F_array,fm);
  this->utils.setFieldData(J_array,fm);
  this->utils.setFieldData(shearModulus,fm);
  this->utils.setFieldData(bulkModulus,fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(hardeningModulus,fm);
  this->utils.setFieldData(yieldStrength,fm);
  this->utils.setFieldData(mechSource,fm);
  this->utils.setFieldData(deltaTime,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  bool print = false;
  if (typeid(ScalarT) == typeid(RealType)) print = true;

  // declare some ScalarT's to be used later
  ScalarT J, Jm23, K, H, Y;
  ScalarT f, dgam;
  ScalarT deltaTemp;
  ScalarT mu, mubar;
  ScalarT smag;
  ScalarT pressure;
  ScalarT sq23 = std::sqrt(2./3.);

  // grab the time step
  ScalarT dt = deltaTime(0);

  // get old state variables
  Albany::MDArray Fpold_array   = (*workset.stateArrayPtr)[fpName];
  Albany::MDArray eqpsold       = (*workset.stateArrayPtr)[eqpsName];

  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      // Fill in tensors from MDArray data
      for (std::size_t i=0; i < numDims; ++i)
      {
	for (std::size_t j=0; j < numDims; ++j)
	{
	  Fpold(i,j) = Fpold_array(cell,qp,i,j);
	  F(i,j)     = F_array(cell,qp,i,j);
	}
      }

      // local qp values (for readibility)
      J     = J_array(cell,qp);
      Jm23  = std::pow(J, -2./3.);
      mu    = shearModulus(cell,qp);
      K     = bulkModulus(cell,qp);
      H     = hardeningModulus(cell,qp);
      Y     = yieldStrength(cell,qp);
      deltaTemp = temperature(cell,qp) - refTemperature;

      // compute the pressure
      pressure = 0.5 * K * ( J - 1/J )
	- 3 * thermalExpansionCoeff * deltaTemp * ( 1 + 1 / ( J * J ) );
      
      // compute trial intermediate configuration
      Fpinv = inverse(Fpold);
      Cpinv = Fpinv*transpose(Fpinv);
      be = F*Cpinv*transpose(F);

      // compute the trial deviatoric stress
      mubar = ScalarT(trace(be)/3.) * mu;
      s = mu * dev(be);

      // check for yielding
      smag = norm(s);
      f = smag - sq23 * ( Y + H * eqpsold(cell,qp) );
     
      if (f > 1E-8)
      {
        // return mapping algorithm
        bool converged = false;
        ScalarT g = f;
        ScalarT G = H * eqpsold(cell,qp);
        ScalarT dg = ( -2. * mubar ) * ( 1. + H / ( 3. * mubar ) );
        ScalarT dG = 0.0;;
        ScalarT alpha = 0.0;
        ScalarT res = 0.0;
        int count = 0;
        dgam = 0.0;

        while (!converged)
        {
          count++;

          dgam -= g/dg;

          alpha = eqpsold(cell,qp) + sq23 * dgam;

          G = H * alpha;
          dG = H;

          g = smag -  ( 2. * mubar * dgam + sq23 * ( Y + G ) );
          dg = -2. * mubar * ( 1. + dG / ( 3. * mubar ) );

          res = std::abs(g);
          if ( res < 1.e-8 || res/f < 1.e-8 )
            converged = true;

          TEUCHOS_TEST_FOR_EXCEPTION( count > 50, std::runtime_error,
                                      std::endl << "Error in return mapping, count = " << count <<
				      "\nres = " << res <<
				      "\nrelres = " << res/f <<
				      "\ng = " << g <<
				      "\ndg = " << dg <<
				      "\nalpha = " << alpha << std::endl);

	}
	
	// plastic direction
	N = ScalarT(1/smag) * s;

	// updated deviatoric stress
	s -= ScalarT(2. * mubar * dgam) * N;

	// update eqps
	eqps(cell,qp) = alpha;

	// exponential map to get Fp
	A = dgam * N;
	expA = LCM::exp<ScalarT>(A);

	for (std::size_t i=0; i < numDims; ++i)	
	{
	  for (std::size_t j=0; j < numDims; ++j)
	  {
	    Fp(cell,qp,i,j) = 0.0;
	    for (std::size_t p=0; p < numDims; ++p)
	    {
	      Fp(cell,qp,i,j) += expA(i,p) * Fpold(p,j);
	    }
	  }
	}
      } 
      else
      {
	// set state variables to old values
	//dp(cell, qp) = 0.0;
	eqps(cell, qp) = eqpsold(cell,qp);
	for (std::size_t i=0; i < numDims; ++i)	
	  for (std::size_t j=0; j < numDims; ++j)
	    Fp(cell,qp,i,j) = Fpold_array(cell,qp,i,j);
      }

      // compute stress
      for (std::size_t i=0; i < numDims; ++i)	
      {
	for (std::size_t j=0; j < numDims; ++j)
	{
	  stress(cell,qp,i,j) = s(i,j) / J;
	}
	stress(cell,qp,i,i) += pressure;
      }

      // update be
      be = ScalarT(1/mu)*s + ScalarT(trace(be)/3)*eye<ScalarT>();

      // plastic work
      mechSource(cell,qp) = dgam*norm(s);

      if (print)
      {
        cout << "********" << endl;
	cout << "work   : " << mechSource(cell,qp) << endl;
	cout << "stress : ";
	for (std::size_t i=0; i < numDims; ++i)	
	  for (std::size_t j=0; j < numDims; ++j)	
	    cout << stress(cell,qp,i,j) << " ";
	cout << endl;
      }
    }
  }
}
//**********************************************************************

}
