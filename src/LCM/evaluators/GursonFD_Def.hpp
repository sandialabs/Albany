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
  f0               (p.get<RealType>("f0 Name"))
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
  ScalarT smag, Phi, p, dgam(0.0);
  ScalarT sq23 = std::sqrt(2./3.);

  ScalarT fvoid, eq;
  LCM::Tensor<ScalarT> s;

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

//      std::cout << "E " << elasticModulus(cell,qp) << std::endl;
//      std::cout << "nu " << poissonsRatio(cell,qp) << std::endl;
//      std::cout << "K" << K << std::endl;
//      std::cout << "Y" << Y << std::endl;
//      std::cout << "siginf " << siginf << std::endl;
//      std::cout << "delta " << delta << std::endl;
//
//      std::cout << "defgrad(cell, qp,1,1)" << defgrad(cell, qp, 1,1) << std::endl;
//
//
//      std::cout << "trace(logbe)/3 " << trd3 << std::endl;
      std::cout << "J " << J(cell,qp) << std::endl;
//      std::cout << "det(be) " << detbe << std::endl;
//      std::cout << "trace(logbe)/3 " << trd3 << std::endl;


      s = mu * (logbe - trd3 * LCM::identity<ScalarT>());
      p = 0.5 * kappa * std::log(detbe);
      fvoid = voidVolumeold(cell,qp);
      eq = eqpsold(cell,qp);

      Phi = compute_Phi(s, p, fvoid, eq, K, Y, siginf, delta);
      std::cout << "Phi= " << Phi << std::endl;
      std::cout << "p= " << p << std::endl;
      std::cout << "fvoid= " << fvoid << std::endl;
      std::cout << "eq= " << eq << std::endl;


      if (Phi > 1.e-12)
      {// plastic yielding

    	  bool converged = false;
    	  int iter = 0;
    	  ScalarT normR0 = 0.0, conv = 0.0, normR = 0.0;
    	  std::vector<ScalarT> R(4);
    	  std::vector<ScalarT> dRdX(16);
    	  std::vector<ScalarT> X(4);

    	  X[0] = p, X[1] = fvoid, X[2] = eq, X[3] = dgam;
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

			//std::cout << "iter= " << iter << std::endl;
			//std::cout << "conv= " << conv << " normR= " << normR << std::endl;

			if(conv < 1.e-12 || normR < 1.e-12) break;
			if(iter > 20) break;


//			std::cout << "before solve " << iter << std::endl;
//	        std::cout << "R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;
//	        std::cout << "X= " << X[0] << " " << X[1] << " " << X[2] << " " << X[3]<< std::endl;
//	        std::cout << "dRdX0, 1, 2, 3= " << dRdX[0] << " "<< dRdX[1] << " "
//	        		<< dRdX[2] << " "<< dRdX[3] << std::endl;
//	        std::cout << "dRdX4, 5, 6, 7= " << dRdX[4] << " "<< dRdX[5] << " "
//	        		<< dRdX[6] << " "<< dRdX[7] << std::endl;
//	        std::cout << "dRdX8, 9, 10, 11= " << dRdX[8] << " "<< dRdX[9] << " "
//	        		<< dRdX[10] << " "<< dRdX[11] << std::endl;
//	        std::cout << "dRdX12, 13, 14, 15= " << dRdX[12] << " "<< dRdX[13] << " "
//	        		<< dRdX[14] << " "<< dRdX[15] << std::endl;

			solver.solve(dRdX, X, R);

//			std::cout << "after solve " << iter << std::endl;
//	        std::cout << "R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;
//	        std::cout << "X= " << X[0] << " " << X[1] << " " << X[2] << " " << X[3]<< std::endl;

			iter ++;
    	  }// end of local N-R

//    	  std::cout << "after local N-R"<< std::endl;
//          std::cout << "R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;
//          std::cout << "X= " << X[0] << " " << X[1] << " " << X[2] << " " << X[3]<< std::endl;
//			std::cout << "dRdX0, 1, 2, 3= " << dRdX[0] << " "<< dRdX[1] << " "
//					<< dRdX[2] << " "<< dRdX[3] << std::endl;
//			std::cout << "dRdX4, 5, 6, 7= " << dRdX[4] << " "<< dRdX[5] << " "
//					<< dRdX[6] << " "<< dRdX[7] << std::endl;
//			std::cout << "dRdX8, 9, 10, 11= " << dRdX[8] << " "<< dRdX[9] << " "
//					<< dRdX[10] << " "<< dRdX[11] << std::endl;
//			std::cout << "dRdX12, 13, 14, 15= " << dRdX[12] << " "<< dRdX[13] << " "
//					<< dRdX[14] << " "<< dRdX[15] << std::endl;
//			std::cout << "=============Calling computeFadInfo ============" << std::endl;

    	  // compute sensitivity information, and pack back to X
          solver.computeFadInfo(dRdX, X, R);

//			std::cout << "=========================" << std::endl;

//    	  std::cout << "after sensitivity information calculated"<< std::endl;
//          std::cout << "R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;
//          std::cout << "X= " << X[0] << " " << X[1] << " " << X[2] << " " << X[3]<< std::endl;

          // update
          p = X[0];
          fvoid = X[1];
          eq = X[2];
          dgam = X[3];
          s = ScalarT(1. / (1. + 2. * mu * dgam)) * s;

      	  ScalarT h = siginf * (1. - std::exp(-delta * eq)) + K * eq;
      	  ScalarT Ybar = Y;
      	  if(std::abs(fvoid - 1.) > 1.0e-12)
      		Ybar = Y + h / (1. - fvoid);

      	  ScalarT tmp = 1.5 * p / Ybar;
      	  ScalarT sinhtmp = std::sinh(tmp);

  		  LCM::Tensor<ScalarT> dPhidtau(0.0);

  		  for (std::size_t i=0; i < numDims; ++i){
			for (std::size_t j=0; j < numDims; ++j){
				dPhidtau(i,j) = s(i,j);
			}
			dPhidtau(i,i) += 1./3. * Ybar * fvoid * sinhtmp;
  		  }

  		  LCM::Tensor<ScalarT> A = dgam * dPhidtau;
  		  LCM::Tensor<ScalarT> expA = LCM::exp(A);

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

      // compute stress (note p also has to be divided by J, since its the Kirchhoff pressure)
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
	ScalarT Ybar = Y;
	if(std::abs(fvoid - 1.) > 1.0e-12)
		Ybar = Y + h / (1. - fvoid);

	ScalarT tmp = 1.5 * p / Ybar;

	ScalarT psi = 1. + fvoid * fvoid - 2. * fvoid * std::cosh(tmp);

	return 0.5 * LCM::dotdot(s, s) - psi * Ybar * Ybar / 3.0;
}

template<typename EvalT, typename Traits>
void
GursonFD<EvalT, Traits>::compute_ResidJacobian(std::vector<ScalarT> & X, std::vector<ScalarT> & R,
		std::vector<ScalarT> & dRdX, const ScalarT & p, const ScalarT & fvoid, const ScalarT & eq,
		LCM::Tensor<ScalarT> & s,ScalarT & mu,ScalarT & kappa,
		ScalarT & K, ScalarT & Y, ScalarT & siginf,	ScalarT & delta)
{
	std::vector<DFadType> Rfad(4);
	std::vector<DFadType> Xfad(4);
	for (std::size_t i=0; i < 4; ++i)
		Xfad[i] = DFadType(4, i, X[i]);

	DFadType pfad = Xfad[0], fvoidfad = Xfad[1], eqfad = Xfad[2], dgam = Xfad[3];

//	std::cout << "=================" << std::endl;
//
//	std::cout << "in assemble Residual" << std::endl;
//    std::cout << "X= " << X[0] << " " << X[1] << " " << X[2] << " " << X[3]<< std::endl;
//    std::cout << "Xfad= " << Xfad[0] << " " << Xfad[1] << " " << Xfad[2] << " " << Xfad[3]<< std::endl;

	// I have to do these step by step, otherwise, it causes compile error
	// Qchen 4/17/12
	DFadType fac; // fac = (1./ (1. + 2. * mu * dgam));
	fac = mu * dgam;
	fac = 1. / (1. + 2. * fac);

//	DFadType fac = (1. / (1. + 2. * mu * dgam));
//    std::cout << "fac= " << fac << std::endl;

	LCM::Tensor<DFadType> sfad(0.0);
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			sfad(i,j) = fac *s(i,j);

	DFadType h; // h = siginf * (1. - std::exp(-delta*eqfad)) + K * eqfad;
	h = delta * eqfad;
	h = -1. * h;
	h = std::exp(h);
	h = 1. - h;
	h = siginf * h;
	h = h + K * eqfad;
//    std::cout << "h= " << h<< std::endl;


	DFadType Ybar = Y;
    if(std::abs(fvoidfad - 1.) > 1.0e-12)
		Ybar = Y + h / (1. - fvoidfad);
//    std::cout << "Ybar= " << Ybar << std::endl;

	DFadType tmp = pfad / Ybar; // tmp = 1.5 * p / Ybar;
	tmp = 1.5 * tmp;
//    std::cout << "tmp= " << tmp << std::endl;

	DFadType fvoid2;//psi = 1. + fvoid * fvoid - 2. * fvoid * std::cosh(tmp);
	fvoid2 = fvoidfad * fvoidfad;

	DFadType psi;
	psi = std::cosh(tmp);
	psi = fvoidfad * psi;
	psi = 2. * psi;
	psi = fvoid2 - psi;
	psi = 1. + psi;
//    std::cout << "psi= " << psi << std::endl;

	DFadType Phi;
	Phi = 0.5 * LCM::dotdot(sfad, sfad) - psi * Ybar * Ybar / 3.0;
//    std::cout << "Phi= " << Phi << std::endl;

	// residual
	Rfad[0] = Phi;
	Rfad[1] = pfad - p + dgam * kappa * Ybar * fvoidfad * std::sinh(tmp);
	Rfad[2] = fvoidfad - fvoid - dgam * fvoidfad * (1. - fvoidfad) * Ybar * std::sinh(tmp);
	Rfad[3] = eqfad - eq - dgam / (1. - fvoidfad) * (2./3. * psi * Ybar - pfad * fvoidfad * std::sinh(tmp));

//    std::cout << "in assemble_Resid, Rfad= " << Rfad[0] << " " << Rfad[1] << " "
//    		<< Rfad[2] << " " << Rfad[3]<< std::endl;

//    std::cout << "in R[3], eqfad & eq = "<< eqfad << " " << eq << std::endl;
//    std::cout << "in R[3], dgam & fvoidfad = "<< dgam << " " << fvoidfad << std::endl;
//    std::cout << "in R[3], psi & Ybar = "<< psi << " " << Ybar << std::endl;
//    std::cout << "in R[3], pfad & sinh(tmp) = "<< pfad << " " << std::sinh(tmp) << std::endl;




	// get ScalarT Residual
	for (int i=0; i<4; i++)
		R[i] = Rfad[i].val();

    //std::cout << "in assemble_Resid, R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;

	// get Jacobian
	for (int i=0; i<4; i++)
		for (int j=0; j<4; j++)
			dRdX[i + 4*j] = Rfad[i].dx(j);
			//dRdX[4*i + j] = Rfad[i].dx(j);

//	std::cout << "---------------" << std::endl;

}

//**********************************************************************
} // end LCM
