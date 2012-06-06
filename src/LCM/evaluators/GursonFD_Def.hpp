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
#include "LocalNonlinearSolver.h"
//#include "Sacado_MathFunctions.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
GursonFD<EvalT, Traits>::
GursonFD(const Teuchos::ParameterList& p) :
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
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  voidVolume       (p.get<std::string>                   ("Void Volume Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  f0               (p.get<RealType>("f0 Name")),
  kw               (p.get<RealType>("kw Name")),
  eN               (p.get<RealType>("eN Name")),
  sN               (p.get<RealType>("sN Name")),
  fN               (p.get<RealType>("fN Name")),
  fc          	   (p.get<RealType>("fc Name")),
  ff          	   (p.get<RealType>("ff Name")),
  q1          	   (p.get<RealType>("q1 Name")),
  q2       	   	   (p.get<RealType>("q2 Name")),
  q3          	   (p.get<RealType>("q3 Name"))
{
    // Pull out numQPs and numDims from a Layout
    Teuchos::RCP<PHX::DataLayout> tensor_dl =
  	p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    tensor_dl->dimensions(dims);
    numQPs  = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(elasticModulus);
    // PoissonRatio not used in 1D stress calc
    if (numDims>1) this->addDependentField(poissonsRatio);
    this->addDependentField(defgrad);
    this->addDependentField(J);
    this->addDependentField(yieldStrength);
    this->addDependentField(hardeningModulus);
    this->addDependentField(satMod);
    this->addDependentField(satExp);

    // state variable
    fpName = p.get<std::string>("Fp Name")+"_old";
    eqpsName = p.get<std::string>("Eqps Name")+"_old";
    voidVolumeName = p.get<std::string>("Void Volume Name")+"_old";

    // evaluated fields
    this->addEvaluatedField(stress);
    this->addEvaluatedField(Fp);
    this->addEvaluatedField(eqps);
    this->addEvaluatedField(voidVolume);

    // scratch space FCs
    Fpinv.resize(worksetSize, numQPs, numDims, numDims);
    FpinvT.resize(worksetSize, numQPs, numDims, numDims);
    Cpinv.resize(worksetSize, numQPs, numDims, numDims);

    this->setName("Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void GursonFD<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(elasticModulus,fm);
  if (numDims>1) this->utils.setFieldData(poissonsRatio,fm);
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(hardeningModulus,fm);
  this->utils.setFieldData(yieldStrength,fm);
  this->utils.setFieldData(satMod,fm);
  this->utils.setFieldData(satExp,fm);
  this->utils.setFieldData(Fp,fm);
  this->utils.setFieldData(eqps,fm);
  this->utils.setFieldData(voidVolume,fm);

}

//**********************************************************************

template<typename EvalT, typename Traits>
void GursonFD<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;
  typedef Intrepid::RealSpaceTools<ScalarT> RST;

  ScalarT kappa;
  ScalarT mu;
  ScalarT K, Y, siginf, delta;
  ScalarT trd3;
  ScalarT Phi, p, dgam(0.0);
  ScalarT sq23 = std::sqrt(2./3.);
  ScalarT sq32 = std::sqrt(3./2.);

  ScalarT fvoid, eq;

  // previous state
  Albany::MDArray Fpold = (*workset.stateArrayPtr)[fpName];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqpsName];
  Albany::MDArray voidVolumeold = (*workset.stateArrayPtr)[voidVolumeName];

  //  // compute Cp_{n}^{-1}
  //  // AGS MAY NEED TO ALLICATE Fpinv FpinvT Cpinv  with actual workse size
  //  // to prevent going past the end of Fpold.
  RST::inverse(Fpinv, Fpold);
  RST::transpose(FpinvT, Fpinv);
  FST::tensorMultiplyDataData<ScalarT>(Cpinv, Fpinv, FpinvT);

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

      // Compute Trial State
      be.clear();
      for (std::size_t i=0; i < numDims; ++i)
        for (std::size_t j=0; j < numDims; ++j)
          for (std::size_t p=0; p < numDims; ++p)
            for (std::size_t q=0; q < numDims; ++q)
              be(i,j) += defgrad(cell,qp,i,p) * Cpinv(cell,qp,p,q) * defgrad(cell,qp,j,q);

      logbe = LCM::log<ScalarT>(be);
      trd3 = LCM::trace(logbe) / 3.0;

      ScalarT detbe = LCM::det<ScalarT>(be);

      s = mu * (logbe - trd3 * LCM::identity<ScalarT>());
      p = 0.5 * kappa * std::log(detbe);
      fvoid = voidVolumeold(cell,qp);
      eq = eqpsold(cell,qp);

      Phi = compute_Phi(s, p, fvoid, eq, K, Y, siginf, delta);

      ScalarT smag2 = LCM::dotdot(s, s);
      ScalarT smag = std::sqrt(smag2);

      if (Phi > 1.e-12)
      {// plastic yielding

    	  bool converged = false;
    	  int iter = 0;
    	  ScalarT normR0 = 0.0, conv = 0.0, normR = 0.0;
    	  std::vector<ScalarT> R(4);
    	  std::vector<ScalarT> dRdX(16);
    	  std::vector<ScalarT> X(4);

    	  dgam = 0.0;

    	  // initialize local unknown vector
    	  X[0] = dgam;
    	  X[1] = p;
    	  X[2] = fvoid;
    	  X[3] = eq;

    	  LocalNonlinearSolver<EvalT, Traits> solver;

    	  // local N-R loop
    	  while (!converged){

    		compute_ResidJacobian(X, R, dRdX, p, fvoid, eq, s, mu, kappa, K, Y, siginf, delta);

    		normR = 0.0;
  			for (int i = 0; i < 4; i++)
  				normR += R[i]*R[i];

  			normR = std::sqrt(normR);

			if(iter == 0) normR0 = normR;
			if(normR0 != 0) conv = normR / normR0;
			else conv = normR0;

			//std::cout << "Iter = " << iter << "conv = " << conv << std::endl;
			if(conv < 1.e-11 || normR < 1.e-11) break;
			if(iter > 20) break;

			solver.solve(dRdX, X, R);

			iter ++;
    	  }// end of local N-R

    	  // compute sensitivity information w.r.t system parameters, and pack back to X
          solver.computeFadInfo(dRdX, X, R);

          // update
          dgam = X[0];
          p = X[1];
          fvoid = X[2];
          eq = X[3];

          for (std::size_t i=0; i < numDims; ++i)
            for (std::size_t j=0; j < numDims; ++j)
              s(i,j) = (1./(1. + 2. * mu * dgam)) * s(i,j);

		  ScalarT h = siginf * (1. - std::exp(-delta * eq)) + K * eq;
		  ScalarT Ybar = Y + h;
		  ScalarT tmp = 1.5 * q2 * p / Ybar;

		  //ScalarT psi;
		  //psi = 1. + q3 * fvoid * fvoid  -  2. * q1 * fvoid * std::cosh(tmp);

  		  LCM::Tensor<ScalarT> dPhi(0.0);

  		  for (std::size_t i=0; i < numDims; ++i){
			for (std::size_t j=0; j < numDims; ++j){
				dPhi(i,j) = s(i,j);
			}
			dPhi(i,i) += 1./3. * q1 * q2 * Ybar * fvoid * std::sinh(tmp);
  		  }

  		  A = dgam * dPhi;
  		  expA = LCM::exp(A);

  		  eqps(cell, qp) = eq;
  		  voidVolume(cell, qp) = fvoid;

          for (std::size_t i=0; i < numDims; ++i){
            for (std::size_t j=0; j < numDims; ++j){
              Fp(cell,qp,i,j) = 0.0;
              for (std::size_t p=0; p < numDims; ++p){
                Fp(cell,qp,i,j) += expA(i,p) * Fpold(cell,qp,p,j);
              }
            }
          }

      } // end of plastic loading
      else
      {// elasticity, set state variables to old values
    	 eqps(cell, qp) = eqpsold(cell,qp);
         voidVolume(cell, qp) = voidVolumeold(cell,qp);
         for (std::size_t i=0; i < numDims; ++i)
           for (std::size_t j=0; j < numDims; ++j)
             Fp(cell,qp,i,j) = Fpold(cell,qp,i,j);

      }

      // compute Cauchy stress tensor
      // (note that p also has to be divided by J, since its the Kirchhoff pressure)
      for (std::size_t i=0; i < numDims; ++i)
      {
        for (std::size_t j=0; j < numDims; ++j)
        {
          stress(cell,qp,i,j) = s(i,j) / J(cell,qp);
        }
        stress(cell,qp,i,i) += p / J(cell,qp);
      }

    }// end of loop over qp
  } //end of loop over cell

  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable
  // values. Leaving this out leads to inversion of 0 tensors.
  for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
      for (std::size_t i=0; i < numDims; ++i)
          Fp(cell,qp,i,i) = 1.0;

} // end of evaluateFields


//**********************************************************************
// all local functions
template<typename EvalT, typename Traits>
typename EvalT::ScalarT
GursonFD<EvalT, Traits>::compute_Phi(LCM::Tensor<ScalarT> & s, ScalarT & p, ScalarT & fvoid,
		ScalarT & eq, ScalarT & K, ScalarT & Y, ScalarT & siginf, ScalarT & delta)
{

	ScalarT h = siginf * (1. - std::exp(-delta * eq)) + K * eq;

	ScalarT Ybar = Y + h;

	ScalarT tmp = 1.5 * q2 * p / Ybar;

	ScalarT psi = 1. + q3 * fvoid * fvoid - 2. * q1 * fvoid * std::cosh(tmp);

	// a quadratic representation will look like:
	ScalarT Phi = 0.5 * LCM::dotdot(s, s) - psi * Ybar * Ybar / 3.0;

	// linear form
//	ScalarT smag = LCM::dotdot(s,s);
//	smag = std::sqrt(smag);
//	ScalarT sq23 = std::sqrt(2./3.);
//  ScalarT Phi = smag - sq23 * std::sqrt(psi) * psi_sign * Ybar

	return Phi;
}

template<typename EvalT, typename Traits>
void
GursonFD<EvalT, Traits>::compute_ResidJacobian(std::vector<ScalarT> & X, std::vector<ScalarT> & R,
		std::vector<ScalarT> & dRdX, const ScalarT & p, const ScalarT & fvoid, const ScalarT & eq,
		LCM::Tensor<ScalarT> & s,ScalarT & mu,ScalarT & kappa,
		ScalarT & K, ScalarT & Y, ScalarT & siginf,	ScalarT & delta)
{
	ScalarT sq32 = std::sqrt(3./2.);
	ScalarT sq23 = std::sqrt(2./3.);
	std::vector<DFadType> Rfad(4);
	std::vector<DFadType> Xfad(4);

	// initialize DFadType local unknown vector Xfad
	// Note that since Xfad is a temporary variable that gets changed within local iterations
	// when we initialize Xfad, we only pass in the values of X, NOT the system sensitivity information
	std::vector<ScalarT> Xval(4);
	for (std::size_t i=0; i < 4; ++i){
		Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
		Xfad[i] = DFadType(4, i, Xval[i]);
	}

	DFadType dgam = Xfad[0], pfad = Xfad[1], fvoidfad = Xfad[2], eqfad = Xfad[3];

	// have to break down these equations, otherwise I get compile error
	DFadType fac;
	fac = mu * dgam;
	fac = 2. * fac;
	fac = 1. + fac;
	fac = 1./fac;

	DFadType h(0.0); // h = siginf * (1. - std::exp(-delta*eqfad)) + K * eqfad;
	h = delta * eqfad;
	h = -1. * h;
	h = std::exp(h);
	h = 1. - h;
	h = siginf * h;
	h = h + K * eqfad;

	DFadType Ybar = Y + h;

	DFadType tmp = pfad / Ybar;
	tmp = 1.5 * tmp;
	tmp = q2 * tmp;

	DFadType fvoid2;
	fvoid2 = fvoidfad * fvoidfad;
	fvoid2 = q3 * fvoid2;

	DFadType psi;
	psi = std::cosh(tmp);
	psi = fvoidfad * psi;
	psi = 2. * psi;
	psi = q1 * psi;
	psi = fvoid2 - psi;
	psi = 1. + psi;

	LCM::Tensor<DFadType> sfad(0.0);

	// valid for assumption Ntr = N;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			sfad(i,j) = fac * s(i,j);
		}
	}
	// shear-dependent term in void growth
	DFadType omega(0.0), J3(0.0), taue(0.0), smag2, smag;
	J3 = LCM::det(sfad);
	smag2 = LCM::dotdot(sfad,sfad);
	if(smag2 > 0){
		smag = std::sqrt(smag2);
		taue = sq32 * smag;
	}

	if(taue > 0)
		omega = 1. - (27. * J3 / 2./taue/taue/taue) * (27. * J3 / 2./taue/taue/taue);

	DFadType deq(0.0);
	deq = dgam * (smag2 + q1 * q2 * pfad * Ybar * fvoidfad * std::sinh(tmp)) / (1. - fvoidfad) / Ybar;

	// void nucleation to be added
	DFadType dfn(0.0);
	DFadType An(0.0), eratio(0.0);
	eratio = -0.5 * (eqfad - eN) * (eqfad - eN) / sN / sN;

	const double pi = acos(-1.0);
	if(pfad >= 0)
		An = fN / sN / (std::sqrt(2.0*pi)) * std::exp(eratio);

	dfn = An * deq;

	DFadType dfg(0.0);
	if(taue > 0){
		dfg = dgam * q1 * q2 * (1. - fvoidfad) * fvoidfad * Ybar * std::sinh(tmp) + sq23 * dgam * kw * fvoidfad * omega * smag;
	}
	else{
		dfg = dgam * q1 * q2 * (1. - fvoidfad) * fvoidfad * Ybar * std::sinh(tmp);
	}


	DFadType Phi;

	Phi = 0.5 * smag2 - psi * Ybar * Ybar / 3.0;

	// local system of equations
	Rfad[0] = Phi;
	Rfad[1] = pfad - p + dgam * q1 * q2 * kappa * Ybar * fvoidfad * std::sinh(tmp);
	Rfad[2] = fvoidfad - fvoid  - dfg - dfn;
	Rfad[3] = eqfad - eq - deq;

	// get ScalarT Residual
	for (int i=0; i<4; i++)
		R[i] = Rfad[i].val();

	// get local Jacobian
	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++)
			dRdX[i + 4*j] = Rfad[i].dx(j);

}

//**********************************************************************
} // end LCM
