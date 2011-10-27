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
J2Stress<EvalT, Traits>::
J2Stress(const Teuchos::ParameterList& p) :
  defgrad          (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J                (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  elasticModulus   (p.get<std::string>                   ("Elastic Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatio    (p.get<std::string>                   ("Poissons Ratio Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  yieldStrength    (p.get<std::string>                   ("Yield Strength Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  hardeningModulus (p.get<std::string>                   ("Hardening Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  satMod           (p.get<std::string>                   ("Saturation Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  satExp           (p.get<std::string>                   ("Saturation Exponent Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  Fp               (p.get<std::string>                   ("Fp Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  eqps             (p.get<std::string>                   ("Eqps Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  worksetSize = dims[0];

  this->addDependentField(defgrad);
  this->addDependentField(J);
  this->addDependentField(elasticModulus);
  this->addDependentField(poissonsRatio);
  this->addDependentField(yieldStrength);
  this->addDependentField(hardeningModulus);
  this->addDependentField(satMod);  
  this->addDependentField(satExp);
  // PoissonRatio not used in 1D stress calc
  //  if (numDims>1) this->addDependentField(poissonsRatio);

  fpName = p.get<std::string>("Fp Name")+"_old";
  eqpsName = p.get<std::string>("Eqps Name")+"_old";
  this->addEvaluatedField(stress);
  this->addEvaluatedField(Fp);
  this->addEvaluatedField(eqps);

  // scratch space FCs
  be.resize(numDims, numDims);
  s.resize(numDims, numDims);
  N.resize(numDims, numDims);
  A.resize(numDims, numDims);
  expA.resize(numDims, numDims);
  Fpinv.resize(worksetSize, numQPs, numDims, numDims);
  FpinvT.resize(worksetSize, numQPs, numDims, numDims);
  Cpinv.resize(worksetSize, numQPs, numDims, numDims);
  tmp.resize(numDims, numDims);
  tmp2.resize(numDims, numDims);

  this->setName("Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void J2Stress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(elasticModulus,fm);
  this->utils.setFieldData(hardeningModulus,fm);
  this->utils.setFieldData(yieldStrength,fm);
  this->utils.setFieldData(satMod,fm);
  this->utils.setFieldData(satExp,fm);
  this->utils.setFieldData(Fp,fm);
  this->utils.setFieldData(eqps,fm);
  if (numDims>1) this->utils.setFieldData(poissonsRatio,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void J2Stress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;
  typedef Intrepid::RealSpaceTools<ScalarT> RST;

  ScalarT kappa;
  ScalarT mu, mubar;
  ScalarT K, Y, siginf, delta;
  ScalarT Jm23;
  ScalarT trace;
  ScalarT smag2, smag, f, p, dgam;
  ScalarT sq23 = std::sqrt(2./3.);

  //Albany::StateVariables  oldState = *workset.oldState;
  //Intrepid::FieldContainer<RealType>& Fpold   = *oldState[fpName];
  //Intrepid::FieldContainer<RealType>& eqpsold = *oldState[eqpsName];

  Albany::MDArray Fpold = (*workset.stateArrayPtr)[fpName];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];

  // compute Cp_{n}^{-1}
  // AGS MAY NEED TO ALLICATE Fpinv FpinvT Cpinv  with actual workse size
  // to prevent going past the end of Fpold.
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
      kappa  = elasticModulus(cell,qp) / ( 3. * ( 1. - 2. * poissonsRatio(cell,qp) ) );
      mu     = elasticModulus(cell,qp) / ( 2. * ( 1. + poissonsRatio(cell,qp) ) );
      K      = hardeningModulus(cell,qp);
      Y      = yieldStrength(cell,qp);
      siginf = satMod(cell,qp);
      delta  = satExp(cell,qp);
      Jm23   = std::pow( J(cell,qp), -2./3. );

      // std::cout << "kappa: " << Sacado::ScalarValue<ScalarT>::eval(kappa) << std::endl;
      // std::cout << "mu   : " << Sacado::ScalarValue<ScalarT>::eval(mu) << std::endl;
      // std::cout << "K    : " << Sacado::ScalarValue<ScalarT>::eval(K) << std::endl;
      // std::cout << "Y    : " << Sacao::ScalarValue<ScalarT>::eval(Y) << std::endl;
      be.initialize(0.0);
      // Compute Trial State      
      for (std::size_t i=0; i < numDims; ++i)
      {	
	for (std::size_t j=0; j < numDims; ++j)
	{
	  for (std::size_t p=0; p < numDims; ++p)
	  {
	    for (std::size_t q=0; q < numDims; ++q)
	    {
	      be(i,j) += Jm23 * defgrad(cell,qp,i,p) * Cpinv(cell,qp,p,q) * defgrad(cell,qp,j,q);
	    }
	  }
	}
      } 
      
      // std::cout << "be: \n" << be;
      
      trace = 0.0;
      for (std::size_t i=0; i < numDims; ++i)
	trace += be(i,i);
      trace /= numDims;
      mubar = trace*mu;
      for (std::size_t i=0; i < numDims; ++i)
      {	
	for (std::size_t j=0; j < numDims; ++j)
	{
	  s(i,j) = mu * be(i,j);
	}
	s(i,i) -= mu * trace;
      }	  
      
      // std::cout << "s: \n" << s;

      // check for yielding
      // smag = s.norm();
      smag2 = 0.0;
      for (std::size_t i=0; i < numDims; ++i)	
	for (std::size_t j=0; j < numDims; ++j)
	  smag2 += s(i,j) * s(i,j);
      smag = std::sqrt(smag2);
      
      f = smag - sq23 * ( Y + K * eqpsold(cell,qp) + siginf * ( 1. - exp( -delta * eqpsold(cell,qp) ) ) );

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
	ScalarT H = K * eqpsold(cell,qp) + siginf*( 1. - exp( -delta * eqpsold(cell,qp) ) );
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

	  H = K * alpha + siginf*( 1. - exp( -delta * alpha ) );
	  dH = K + delta * siginf * exp( -delta * alpha );

	  g = smag -  ( 2. * mubar * dgam + sq23 * ( Y + H ) );
	  dg = -2. * mubar * ( 1. + dH / ( 3. * mubar ) );

	  res = std::abs(g);
	  if ( res < 1.e-12 || res/f < 1.E-12 )
	    converged = true;

	  TEUCHOS_TEST_FOR_EXCEPTION( count > 20, std::runtime_error,
				      std::endl << "Error in return mapping, count = " << count <<
                                      "\nres = " << res <<
                                      "\nrelres = " << res/f <<
                                      "\ng = " << g <<
                                      "\ndg = " << dg <<
                                      "\nalpha = " << alpha << std::endl);

        }

        // plastic direction
        for (std::size_t i=0; i < numDims; ++i) 
          for (std::size_t j=0; j < numDims; ++j)
            N(i,j) = (1/smag) * s(i,j);

        for (std::size_t i=0; i < numDims; ++i) 
          for (std::size_t j=0; j < numDims; ++j)
            s(i,j) -= 2. * mubar * dgam * N(i,j);

        // update eqps
        //eqps(cell,qp) = eqpsold(cell,qp) + sqrt(2./3.) * dgam;
        eqps(cell,qp) = alpha;

        // exponential map to get Fp
        for (std::size_t i=0; i < numDims; ++i) 
          for (std::size_t j=0; j < numDims; ++j)
            A(i,j) = dgam * N(i,j);

        exponential_map(expA, A);

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

  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to inversion of 0 tensors.
  for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp) 
      for (std::size_t i=0; i < numDims; ++i)   
          Fp(cell,qp,i,i) = 1.0;
}
//**********************************************************************
template<typename EvalT, typename Traits>
void 
J2Stress<EvalT, Traits>::exponential_map(Intrepid::FieldContainer<ScalarT> & expA, const Intrepid::FieldContainer<ScalarT> A)
{
  tmp.initialize(0.0);
  expA.initialize(0.0);

  bool converged = false;
  ScalarT norm0 = norm(A);

  for (std::size_t i=0; i < numDims; ++i)
  {
    tmp(i,i) = 1.0;
  }

  ScalarT k = 0.0;
  while (!converged)
  {
    // expA += tmp
    for (std::size_t i=0; i < numDims; ++i)
      for (std::size_t j=0; j < numDims; ++j)
        expA(i,j) += tmp(i,j);

    tmp2.initialize(0.0);
    for (std::size_t i=0; i < numDims; ++i)
      for (std::size_t j=0; j < numDims; ++j)
        for (std::size_t p=0; p < numDims; ++p)
          tmp2(i,j) += A(i,p) * tmp(p,j);

    // tmp = tmp2
    k = k + 1.0;
    for (std::size_t i=0; i < numDims; ++i)
      for (std::size_t j=0; j < numDims; ++j)
        tmp(i,j) = (1/k) * tmp2(i,j);

    if (norm(tmp)/norm0 < 1.E-14 ) converged = true;
    
    TEUCHOS_TEST_FOR_EXCEPTION( k > 50.0, std::runtime_error,
                                std::endl << "Error in exponential map, k = " << k << 
                                "\nnorm0 = " << norm0 <<
                                "\nnorm = " << norm(tmp)/norm0 <<
                                "\nA = \n" << A << std::endl);
    
  }
}
//**********************************************************************
template<typename EvalT, typename Traits>
typename EvalT::ScalarT 
J2Stress<EvalT, Traits>::norm(Intrepid::FieldContainer<ScalarT> A)
{
  ScalarT max(0.0), colsum;

  for (std::size_t i(0); i < numDims; ++i)
  {
    colsum = 0.0;
    for (std::size_t j(0); j < numDims; ++j)
      colsum += std::abs(A(i,j));
    max = (colsum > max) ? colsum : max;  
  }

  return max;
}
//**********************************************************************
} // end LCM

