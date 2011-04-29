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

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
J2Damage<EvalT, Traits>::
J2Damage(const Teuchos::ParameterList& p) :
  defgrad          (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J                (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  bulkModulus      (p.get<std::string>                   ("Bulk Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  shearModulus     (p.get<std::string>                   ("Shear Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  yieldStrength    (p.get<std::string>                   ("Yield Strength Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  hardeningModulus (p.get<std::string>                   ("Hardening Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  satMod           (p.get<std::string>                   ("Saturation Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  satExp           (p.get<std::string>                   ("Saturation Exponent Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  damage           (p.get<std::string>                   ("Damage Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  dp               (p.get<std::string>                   ("DP Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  seff             (p.get<std::string>                   ("Effective Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  //int worksetSize = dims[0];
  std::size_t worksetSize = dims[0];

  this->addDependentField(defgrad);
  this->addDependentField(J);
  this->addDependentField(bulkModulus);
  this->addDependentField(shearModulus);
  this->addDependentField(yieldStrength);
  this->addDependentField(hardeningModulus);
  this->addDependentField(satMod);  
  this->addDependentField(satExp);
  this->addDependentField(damage);

  this->addEvaluatedField(stress);
  this->addEvaluatedField(dp);
  this->addEvaluatedField(seff);

  // scratch space FCs
  Fp.resize(worksetSize, numQPs, numDims, numDims);
  Fpinv.resize(worksetSize, numQPs, numDims, numDims);
  FpinvT.resize(worksetSize, numQPs, numDims, numDims);
  Cpinv.resize(worksetSize, numQPs, numDims, numDims);
  eqps.resize(worksetSize, numQPs);

  this->setName("Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void J2Damage<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(dp,fm);
  this->utils.setFieldData(seff,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(bulkModulus,fm);
  this->utils.setFieldData(shearModulus,fm);
  this->utils.setFieldData(hardeningModulus,fm);
  this->utils.setFieldData(yieldStrength,fm);
  this->utils.setFieldData(satMod,fm);
  this->utils.setFieldData(satExp,fm);
  this->utils.setFieldData(damage,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void J2Damage<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;
  typedef Intrepid::RealSpaceTools<ScalarT> RST;

  ScalarT kappa;
  ScalarT mu, mubar;
  ScalarT K, Y, siginf, delta;
  ScalarT Jm23;
  ScalarT trd3;
  ScalarT smag, f, p, dgam;
  ScalarT exph;
  ScalarT sq23 = std::sqrt(2./3.);

  //bool saveState = (workset.newState != Teuchos::null);
  //std::cout << "saveState: " << saveState << std::endl;
  //saveState = (workset.oldState != Teuchos::null);
  //std::cout << "saveState: " << saveState << std::endl;
  Albany::StateVariables& newState = *workset.newState;
  Albany::StateVariables  oldState = *workset.oldState;
  Intrepid::FieldContainer<RealType>& Fpold   = *oldState["Fp"];
  Intrepid::FieldContainer<RealType>& eqpsold = *oldState["eqps"];
  Intrepid::FieldContainer<RealType>& FP      = *newState["Fp"];
  Intrepid::FieldContainer<RealType>& EQPS    = *newState["eqps"];
  Intrepid::FieldContainer<RealType>& STRESS  = *newState["stress"];

  // compute Cp_{n}^{-1}
  RST::inverse(Fpinv, Fpold);
  RST::transpose(FpinvT, Fpinv);
  FST::tensorMultiplyDataData<ScalarT>(Cpinv, Fpinv, FpinvT);

  // std::cout << "F:\n";
  // for (std::size_t cell=0; cell < workset.numCells; ++cell)
  // {
  //   for (std::size_t qp=0; qp < numQPs; ++qp)
  //   {
  //     for (std::size_t i=0; i < numDims; ++i)
  // 	for (std::size_t j=0; j < numDims; ++j)
  // 	  std::cout << Sacado::ScalarValue<ScalarT>::eval(defgrad(cell,qp,i,j)) << " ";
  //   }
  //   std::cout << std::endl;      
  // }
  // std::cout << std::endl;      

  // std::cout << "Fpold:\n";
  // for (std::size_t cell=0; cell < workset.numCells; ++cell)
  // {
  //   for (std::size_t qp=0; qp < numQPs; ++qp)
  //   {
  //     for (std::size_t i=0; i < numDims; ++i)
  // 	for (std::size_t j=0; j < numDims; ++j)
  // 	  std::cout << Sacado::ScalarValue<ScalarT>::eval(Fpold(cell,qp,i,j)) << " ";
  //   }
  //   std::cout << std::endl;      
  // }
  // std::cout << std::endl;      
    
  // std::cout << "Cpinv:\n";
  // for (std::size_t cell=0; cell < workset.numCells; ++cell)
  // {
  //   for (std::size_t qp=0; qp < numQPs; ++qp)
  //   {
  //     for (std::size_t i=0; i < numDims; ++i)
  // 	for (std::size_t j=0; j < numDims; ++j)
  // 	  std::cout << Sacado::ScalarValue<ScalarT>::eval(Cpinv(cell,qp,i,j)) << " ";
  //   }
  //   std::cout << std::endl;      
  // }
  // std::cout << std::endl;      


  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      // local parameters
      kappa  = bulkModulus(cell,qp);
      mu     = shearModulus(cell,qp);
      K      = hardeningModulus(cell,qp);
      Y      = yieldStrength(cell,qp);
      siginf = satMod(cell,qp);
      delta  = satExp(cell,qp);
      Jm23   = std::pow( J(cell,qp), -2./3. );
      exph   = std::exp( damage(cell,qp) );

      // std::cout << "kappa: " << Sacado::ScalarValue<ScalarT>::eval(kappa) << std::endl;
      // std::cout << "mu   : " << Sacado::ScalarValue<ScalarT>::eval(mu) << std::endl;
      // std::cout << "K    : " << Sacado::ScalarValue<ScalarT>::eval(K) << std::endl;
      // std::cout << "Y    : " << Sacao::ScalarValue<ScalarT>::eval(Y) << std::endl;
      be.clear();
      // Compute Trial State      
      for (std::size_t i=0; i < numDims; ++i)
	for (std::size_t j=0; j < numDims; ++j)
	  for (std::size_t p=0; p < numDims; ++p)
	    for (std::size_t q=0; q < numDims; ++q)
	      be(i,j) += Jm23 * defgrad(cell,qp,i,p) * Cpinv(cell,qp,p,q) * defgrad(cell,qp,j,q);
      
      // std::cout << "be: \n" << be;
      
      trd3 = trace(be)/3.;
      mubar = trd3*mu;
      s = mu*(be - trd3*eye<ScalarT>());
      
      // std::cout << "s: \n" << s;

      // check for yielding
      smag = norm(s);
      f = smag - exph * sq23 * ( Y + K * eqpsold(cell,qp) + siginf * ( 1. - std::exp( -delta * eqpsold(cell,qp) ) ) );

      // std::cout << "smag : " << Sacado::ScalarValue<ScalarT>::eval(smag) << std::endl;
      // std::cout << "eqpsold: " << Sacado::ScalarValue<ScalarT>::eval(eqpsold(cell,qp)) << std::endl;
      // std::cout << "K      : " << Sacado::ScalarValue<ScalarT>::eval(K) << std::endl;
      // std::cout << "Y      : " << Sacado::ScalarValue<ScalarT>::eval(Y) << std::endl;
      // std::cout << "f      : " << Sacado::ScalarValue<ScalarT>::eval(f) << std::endl;

      if (f > 1E-12)
      {
	// return mapping algorithm
	bool converged = false;
	ScalarT g = f;
	ScalarT H = K * eqpsold(cell,qp) + siginf*( 1. - std::exp( -delta * eqpsold(cell,qp) ) );
	ScalarT dg = ( -2. * mubar ) * ( 1. + H / ( 3. * mubar ) );
	ScalarT dH = 0.0;;
	ScalarT alpha = 0.0;
	ScalarT res = 0.0;
	int count = 0;
	dgam = 0.0;

	while (!converged)
	{
	  count++;

	  //dgam = ( f / ( 2. * mubar) ) / ( 1. + K / ( 3. * mubar ) );
	  dgam -= g/dg;

	  alpha = eqpsold(cell,qp) + sq23 * dgam;

	  H = K * alpha + exph * siginf * ( 1. - std::exp( -delta * alpha ) );
	  dH = K + exph * delta * siginf * std::exp( -delta * alpha );

	  g = smag -  ( 2. * mubar * dgam + sq23 * ( Y + H ) );
	  dg = -2. * mubar * ( 1. + dH / ( 3. * mubar ) );

	  res = std::abs(g);
	  if ( res < 1.e-14 || res/f < 1.E-14 )
	    converged = true;

	  TEST_FOR_EXCEPTION( count > 20, std::runtime_error,
			      std::endl << "Error in return mapping, count = " << count << 			
			      "\nres = " << res <<
			      "\ng = " << g <<
			      "\ndg = " << dg <<
			      "\nalpha = " << alpha << std::endl);

	}
	
	// set dp for this QP
	dp(cell,qp) = dgam;

	// plastic direction
	N = ScalarT(1./smag) * s;

	// updated deviatoric stress
	s -= ScalarT(2. * mubar * dgam) * N;

	// update eqps
	eqps(cell,qp) = alpha;

	// exponential map to get Fp
	A = dgam * N;
	expA = LCM::exp<ScalarT>(A);

	// std::cout << "expA: \n";
	// for (std::size_t i=0; i < numDims; ++i)	
	//   for (std::size_t j=0; j < numDims; ++j)
	//     std::cout << Sacado::ScalarValue<ScalarT>::eval(expA(i,j)) << " ";
	// std::cout << std::endl;
		  
	for (std::size_t i=0; i < numDims; ++i)	
	{
	  for (std::size_t j=0; j < numDims; ++j)
	  {
	    Fp(cell,qp,i,j) = 0.0;
	    for (std::size_t p=0; p < numDims; ++p)
	    {
	      Fp(cell,qp,i,j) += expA(i,p) * Fpold(cell,qp,p,j);
	    }
	  }
	}
      } 
      else
      {
	// set state variables to old values
	eqps(cell, qp) = eqpsold(cell,qp);
	for (std::size_t i=0; i < numDims; ++i)	
	  for (std::size_t j=0; j < numDims; ++j)
	    Fp(cell,qp,i,j) = Fpold(cell,qp,i,j);
      }
      

      // compute pressure
      p = kappa * ( J(cell,qp) - 1 / ( J(cell,qp) ) );
      
      // compute stress
      for (std::size_t i=0; i < numDims; ++i)	
      {
	for (std::size_t j=0; j < numDims; ++j)
	{
	  stress(cell,qp,i,j) = s(i,j) / J(cell,qp);
	}
	stress(cell,qp,i,i) += p;
      }
    }
  }

  // std::cout << "Fp: \n";
  // Save Stress as State Variable
  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      EQPS(cell,qp) = Sacado::ScalarValue<ScalarT>::eval(eqps(cell,qp));
      for (std::size_t i=0; i < numDims; ++i)
      {
	for (std::size_t j=0; j < numDims; ++j)
	{
	  STRESS(cell,qp,i,j) = Sacado::ScalarValue<ScalarT>::eval(stress(cell,qp,i,j));
	  FP(cell,qp,i,j) = Sacado::ScalarValue<ScalarT>::eval(Fp(cell,qp,i,j));
	  // std::cout << FP(cell,qp,i,j) << " ";
	}
      }
      // std::cout << std::endl;
    }
    // std::cout << std::endl;
  }
}
//**********************************************************************
} // end LCM

