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
namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
CapImplicit<EvalT, Traits>::
CapImplicit(const Teuchos::ParameterList& p) :
  elasticModulus   (p.get<std::string>                   ("Elastic Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatio    (p.get<std::string>                   ("Poissons Ratio Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  strain           (p.get<std::string>                   ("Strain Name"),
   	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  backStress       (p.get<std::string>                   ("Back Stress Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  capParameter	   (p.get<std::string>                   ("Cap Parameter Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  A          	   (p.get<RealType>("A Name")),
  B          	   (p.get<RealType>("B Name")),
  C          	   (p.get<RealType>("C Name")),
  theta        	   (p.get<RealType>("Theta Name")),
  R          	   (p.get<RealType>("R Name")),
  kappa0       	   (p.get<RealType>("Kappa0 Name")),
  W          	   (p.get<RealType>("W Name")),
  D1          	   (p.get<RealType>("D1 Name")),
  D2          	   (p.get<RealType>("D2 Name")),
  calpha       	   (p.get<RealType>("Calpha Name")),
  psi          	   (p.get<RealType>("Psi Name")),
  N          	   (p.get<RealType>("N Name")),
  L          	   (p.get<RealType>("L Name")),
  phi          	   (p.get<RealType>("Phi Name")),
  Q          	   (p.get<RealType>("Q Name"))
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
	p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(elasticModulus);
  // PoissonRatio not used in 1D stress calc
  if (numDims>1) this->addDependentField(poissonsRatio);
  this->addDependentField(strain);

  // state variable
  strainName = p.get<std::string>("Strain Name")+"_old";
  stressName = p.get<std::string>("Stress Name")+"_old";
  backStressName = p.get<std::string>("Back Stress Name")+"_old";
  capParameterName = p.get<std::string>("Cap Parameter Name")+"_old";

  // evaluated fields
  this->addEvaluatedField(stress);
  this->addEvaluatedField(backStress);
  this->addEvaluatedField(capParameter);

  this->setName("Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CapImplicit<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(elasticModulus,fm);
  if (numDims>1) this->utils.setFieldData(poissonsRatio,fm);
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(strain,fm);
  this->utils.setFieldData(backStress,fm);
  this->utils.setFieldData(capParameter,fm);
}

//**********************************************************************

template<typename EvalT, typename Traits>
void CapImplicit<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // previous state
  Albany::MDArray strainold = (*workset.stateArrayPtr)[strainName];
  Albany::MDArray stressold = (*workset.stateArrayPtr)[stressName];
  Albany::MDArray backStressold = (*workset.stateArrayPtr)[backStressName];
  Albany::MDArray capParameterold = (*workset.stateArrayPtr)[capParameterName];

  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
    for (std::size_t qp=0; qp < numQPs; ++qp)
    {
      // local parameters
      ScalarT lame = elasticModulus(cell,qp) * poissonsRatio(cell,qp) / ( 1.0 + poissonsRatio(cell,qp) ) / ( 1.0 - 2.0 * poissonsRatio(cell,qp) );
      ScalarT mu   = elasticModulus(cell,qp) / 2.0 / (1.0 + poissonsRatio(cell,qp) );

      // elastic matrix
      LCM::Tensor4<ScalarT> Celastic = lame * LCM::identity_3<ScalarT>()
    		+ mu * (LCM::identity_1<ScalarT>() + LCM::identity_2<ScalarT>());

  	  // incremental strain tensor
  	  LCM::Tensor<ScalarT> depsilon;
  	  for (std::size_t i = 0; i < numDims; ++i)
  	  	  for (std::size_t j = 0; j < numDims; ++j)
  	  		  depsilon(i,j) = strain(cell,qp,i,j) - strainold(cell,qp,i,j);

  	  // trial state
  	  LCM::Tensor<ScalarT> sigmaVal = LCM::dotdot(Celastic, depsilon);
  	  LCM::Tensor<ScalarT> alphaVal;

  	  for (std::size_t i = 0; i < numDims; ++i){
  	  	  for (std::size_t j = 0; j < numDims; ++j){
  	  		  sigmaVal(i,j) = sigmaVal(i,j) + stressold(cell,qp,i,j);
  	  		  alphaVal(i,j) = backStressold(cell,qp,i,j);
  	  	  }
  	  }

  	  ScalarT kappaVal = capParameterold(cell,qp);
  	  ScalarT dgammaVal = 0.0;

  	  std::vector<ScalarT> XXVal(13);

  	  // check yielding
  	  ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);
  	  XXVal  = initialize(sigmaVal, alphaVal, kappaVal, dgammaVal);

  	  // local NW
  	  if (f > 1.e-10)
  	  {// plastic yielding

		  ScalarT normR, normR0, conv;
		  bool kappa_flag = false;
		  bool converged = false;
		  int iter = 0;

		  std::vector<ScalarT> R(13);
		  std::vector<ScalarT> dRdX(13*13);
    	  LocalNonlinearSolver<EvalT, Traits> solver;

		  while (!converged){

			// assemble residual vector and local Jacobian
			compute_ResidJacobian(XXVal, R, dRdX, sigmaVal, alphaVal, kappaVal, Celastic, kappa_flag);

			normR = 0.0;
			for (int i = 0; i < 13; i++)
				normR += R[i]*R[i];

			normR = std::sqrt(normR);

			if(iter == 0) normR0 = normR;
			if(normR0 != 0) conv = normR / normR0;
			else conv = normR0;

//			std::cout << "iter= " << iter << std::endl;
//			std::cout << "conv= " << Sacado::ScalarValue<ScalarT>::eval(conv)
//					<< " normR= " << Sacado::ScalarValue<ScalarT>::eval(normR) << std::endl;

			if(conv < 1.e-10 || normR < 1.e-10) break;
			//if(iter > 20) break;
			TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
						      std::endl << "Error in return mapping, iter = " << iter <<
		                                      "\nres = " << normR <<
		                                      "\nrelres = " << conv << std::endl);

			std::vector<ScalarT> XXValK = XXVal;
			solver.solve(dRdX, XXValK, R);

			// put restrictions on kappa: only allows monotonic decreasing (cap hardening)
			if(XXValK[11] > XXVal[11]){
				kappa_flag = true;
			}
			else {XXVal = XXValK; kappa_flag = false;}

			// debugging
			//XXVal = XXValK;

			iter ++;
		  }//end local NR

	  	  // compute sensitivity information, and pack back to X.
	  	  solver.computeFadInfo(dRdX, XXVal, R);

  	  }// end of plasticity


  	  // update
	  sigmaVal(0,0) = XXVal[0]; sigmaVal(0,1) = XXVal[5]; sigmaVal(0,2) = XXVal[4];
	  sigmaVal(1,0) = XXVal[5]; sigmaVal(1,1) = XXVal[1]; sigmaVal(1,2) = XXVal[3];
	  sigmaVal(2,0) = XXVal[4]; sigmaVal(2,1) = XXVal[3]; sigmaVal(2,2) = XXVal[2];

	  alphaVal(0,0) = XXVal[6];  alphaVal(0,1) = XXVal[10]; alphaVal(0,2) = XXVal[9];
	  alphaVal(1,0) = XXVal[10]; alphaVal(1,1) = XXVal[7];  alphaVal(1,2) = XXVal[8];
	  alphaVal(2,0) = XXVal[9];  alphaVal(2,1) = XXVal[8];  alphaVal(2,2) = -XXVal[6]-XXVal[7];

	  kappaVal = XXVal[11];

	  //dgammaVal = XXVal[12];

	  // update
	  for (std::size_t i = 0; i < numDims; ++i){
	  	  for (std::size_t j = 0; j < numDims; ++j){
	  		  stress(cell,qp,i,j) = sigmaVal(i,j);
	  		  backStress(cell,qp,i,j) = alphaVal(i,j);
	  	  }
	  }

	  capParameter(cell,qp) = kappaVal;

	} //loop over qps

  }//loop over cell


} // end of evaluateFields

//**********************************************************************
// all local functions
template<typename EvalT, typename Traits>
typename CapImplicit<EvalT, Traits>::ScalarT
CapImplicit<EvalT, Traits>::compute_f(LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
{

	LCM::Tensor<ScalarT> xi = sigma - alpha;

	ScalarT I1 = LCM::trace(xi), p = I1 / 3.;

	LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>();

	ScalarT J2 = 0.5 * LCM::dotdot(s, s);

	ScalarT J3 = LCM::det(s);

	ScalarT Gamma = 1.0;
	if (psi!=0 && J2!=0) Gamma = 0.5 * (1. - 3. * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
			+ (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);

	ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

	ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

	ScalarT X = kappa -  R * Ff_kappa;

	ScalarT Fc = 1.0;

	if((kappa - I1) > 0 && ((X - kappa) != 0))
		Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

	return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template<typename EvalT, typename Traits>
typename CapImplicit<EvalT, Traits>::DFadType
CapImplicit<EvalT, Traits>::compute_f(LCM::Tensor<DFadType> & sigma, LCM::Tensor<DFadType> & alpha, DFadType & kappa)
{

	LCM::Tensor<DFadType> xi = sigma - alpha;

	DFadType I1 = LCM::trace(xi), p = I1 / 3.;

	LCM::Tensor<DFadType> s = xi - p * LCM::identity<DFadType>();

	DFadType J2 = 0.5 * LCM::dotdot(s, s);

	DFadType J3 = LCM::det(s);

	DFadType Gamma = 1.0;
	if (psi!=0 && J2!=0) Gamma = 0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
			+ (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);

	DFadType Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

	DFadType Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

	DFadType X = kappa -  R * Ff_kappa;

	DFadType Fc = 1.0;

	if((kappa - I1) > 0 && ((X - kappa) != 0))
		Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

	return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template<typename EvalT, typename Traits>
typename CapImplicit<EvalT, Traits>::D2FadType
CapImplicit<EvalT, Traits>::compute_g(LCM::Tensor<D2FadType> & sigma, LCM::Tensor<D2FadType> & alpha, D2FadType & kappa)
{

	LCM::Tensor<D2FadType> xi = sigma - alpha;

	D2FadType I1 = LCM::trace(xi), p = I1 / 3.;

	LCM::Tensor<D2FadType> s = xi - p * LCM::identity<D2FadType>();

	D2FadType J2 = 0.5 * LCM::dotdot(s, s);

	D2FadType J3 = LCM::det(s);

	D2FadType Gamma = 1.0;
	if (psi!=0 && J2!=0) Gamma = 0.5 * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
			+ (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)) / psi);

	D2FadType Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

	D2FadType Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

	D2FadType X = kappa -  Q * Ff_kappa;

	D2FadType Fc = 1.0;

	if((kappa - I1) > 0 && ((X - kappa) != 0))
		Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

	return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template<typename EvalT, typename Traits>
std::vector<typename CapImplicit<EvalT, Traits>::ScalarT>
CapImplicit<EvalT, Traits>::initialize(LCM::Tensor<ScalarT> & sigmaVal, LCM::Tensor<ScalarT> & alphaVal, ScalarT & kappaVal, ScalarT & dgammaVal)
{
	std::vector<ScalarT> XX(13);

	XX[0]  = sigmaVal(0,0);
	XX[1]  = sigmaVal(1,1);
	XX[2]  = sigmaVal(2,2);
	XX[3]  = sigmaVal(1,2);
	XX[4]  = sigmaVal(0,2);
	XX[5]  = sigmaVal(0,1);
	XX[6]  = alphaVal(0,0);
	XX[7]  = alphaVal(1,1);
	XX[8]  = alphaVal(1,2);
	XX[9]  = alphaVal(0,2);
	XX[10] = alphaVal(0,1);
	XX[11] = kappaVal;
	XX[12] = dgammaVal;

	return XX;
}

template<typename EvalT, typename Traits>
LCM::Tensor<typename CapImplicit<EvalT, Traits>::DFadType>
CapImplicit<EvalT, Traits>::compute_dgdsigma(std::vector<DFadType> const & XX)
{
	std::vector<D2FadType> D2XX(13);

	for (int i = 0; i < 13; ++i){
		D2XX[i] = D2FadType(13, i, XX[i]);
	}

	LCM::Tensor<D2FadType> sigma, alpha;

	sigma(0,0) = D2XX[0]; sigma(0,1) = D2XX[5]; sigma(0,2) = D2XX[4];
	sigma(1,0) = D2XX[5]; sigma(1,1) = D2XX[1]; sigma(1,2) = D2XX[3];
	sigma(2,0) = D2XX[4]; sigma(2,1) = D2XX[3]; sigma(2,2) = D2XX[2];

	alpha(0,0) = D2XX[6];  alpha(0,1) = D2XX[10]; alpha(0,2) = D2XX[9];
	alpha(1,0) = D2XX[10]; alpha(1,1) = D2XX[7];  alpha(1,2) = D2XX[8];
	alpha(2,0) = D2XX[9];  alpha(2,1) = D2XX[8];  alpha(2,2) = -D2XX[6]-D2XX[7];

	D2FadType kappa = D2XX[11];

	D2FadType g = compute_g(sigma, alpha, kappa);

	LCM::Tensor<DFadType> dgdsigma;

	dgdsigma(0,0) = g.dx(0); dgdsigma(0,1) = g.dx(5); dgdsigma(0,2) = g.dx(4);
	dgdsigma(1,0) = g.dx(5); dgdsigma(1,1) = g.dx(1); dgdsigma(1,2) = g.dx(3);
	dgdsigma(2,0) = g.dx(4); dgdsigma(2,1) = g.dx(3); dgdsigma(2,2) = g.dx(2);

	return dgdsigma;
}

template<typename EvalT, typename Traits>
typename CapImplicit<EvalT, Traits>::DFadType
CapImplicit<EvalT, Traits>::compute_Galpha(DFadType J2_alpha)
{
	if(N != 0)
		return 1.0 - pow(J2_alpha, 0.5) / N;
	else
		return 0.0;
}

template<typename EvalT, typename Traits>
LCM::Tensor<typename CapImplicit<EvalT, Traits>::DFadType>
CapImplicit<EvalT, Traits>::compute_halpha(LCM::Tensor<DFadType> const & dgdsigma,
		DFadType const J2_alpha)
{

	DFadType Galpha = compute_Galpha(J2_alpha);

	DFadType I1 = LCM::trace(dgdsigma), p = I1 / 3.0;

	LCM::Tensor<DFadType> s = dgdsigma - p * LCM::identity<DFadType>();

	//LCM::Tensor<DFadType> halpha = calpha * Galpha * s; // * operator not defined;
	LCM::Tensor<DFadType> halpha;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j<3; j++){
			halpha(i,j) = calpha * Galpha * s(i,j);
		}
	}

	return halpha;
}

template<typename EvalT, typename Traits>
typename CapImplicit<EvalT, Traits>::DFadType
CapImplicit<EvalT, Traits>::compute_dedkappa(DFadType const kappa)
{

	//******** use analytical expression
	ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;

	ScalarT X0 = kappa0 -  Q * Ff_kappa0;

	DFadType Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

	DFadType X = kappa -  Q * Ff_kappa;

	DFadType dedX = (D1 - 2. * D2 * (X-X0)) * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

	DFadType dXdkappa = 1. + Q * C * L * exp(L * kappa) + Q * phi;

	return dedX * dXdkappa;
}

template<typename EvalT, typename Traits>
typename CapImplicit<EvalT, Traits>::DFadType
CapImplicit<EvalT, Traits>::compute_hkappa(DFadType const I1_dgdsigma, DFadType const dedkappa)
{
	if(dedkappa!=0)
		return I1_dgdsigma / dedkappa;
	else
		return 0;
}

template<typename EvalT, typename Traits>
void
CapImplicit<EvalT, Traits>::compute_ResidJacobian(std::vector<ScalarT> const & XXVal, std::vector<ScalarT> & R,
		std::vector<ScalarT> & dRdX, const LCM::Tensor<ScalarT> & sigmaVal, const LCM::Tensor<ScalarT> & alphaVal,
		const ScalarT & kappaVal, LCM::Tensor4<ScalarT> const & Celastic, bool kappa_flag)
{

	std::vector<DFadType> Rfad(13);
	std::vector<DFadType> XX(13);
	std::vector<ScalarT> XXtmp(13);

	// initialize DFadType local unknown vector Xfad
	// Note that since Xfad is a temporary variable that gets changed within local iterations
	// when we initialize Xfad, we only pass in the values of X, NOT the system sensitivity information
	for (int i = 0; i < 13; ++i){
		XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XXVal[i]);
		XX[i] = DFadType(13, i, XXtmp[i]);
	}

	LCM::Tensor<DFadType> sigma, alpha;

	sigma(0,0) = XX[0]; sigma(0,1) = XX[5]; sigma(0,2) = XX[4];
	sigma(1,0) = XX[5]; sigma(1,1) = XX[1]; sigma(1,2) = XX[3];
	sigma(2,0) = XX[4]; sigma(2,1) = XX[3]; sigma(2,2) = XX[2];

	alpha(0,0) = XX[6]; alpha(0,1) = XX[10]; alpha(0,2) = XX[9];
	alpha(1,0) = XX[10]; alpha(1,1) = XX[7]; alpha(1,2) = XX[8];
	alpha(2,0) = XX[9]; alpha(2,1) = XX[8]; alpha(2,2) = -XX[6]-XX[7];

	DFadType kappa = XX[11];

	DFadType dgamma = XX[12];

	DFadType f = compute_f(sigma, alpha, kappa);

	LCM::Tensor<DFadType> dgdsigma = compute_dgdsigma(XX);

	DFadType J2_alpha = 0.5 * LCM::dotdot(alpha, alpha);

	LCM::Tensor<DFadType> halpha = compute_halpha(dgdsigma, J2_alpha);

	DFadType I1_dgdsigma = LCM::trace(dgdsigma);

	DFadType dedkappa = compute_dedkappa(kappa);

	DFadType hkappa = compute_hkappa(I1_dgdsigma, dedkappa);

	DFadType t;

	t = 0;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			t = t + Celastic(0,0,i,j) * dgdsigma(i,j);
		}
	}
	Rfad[0] = dgamma * t + sigma(0,0) - sigmaVal(0,0);

	t = 0;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			t = t + Celastic(1,1,i,j) * dgdsigma(i,j);
		}
	}
	Rfad[1] = dgamma * t + sigma(1,1) - sigmaVal(1,1);

	t = 0;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			t = t + Celastic(2,2,i,j) * dgdsigma(i,j);
		}
	}
	Rfad[2] = dgamma * t + sigma(2,2) - sigmaVal(2,2);

	t = 0;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			t = t + Celastic(1,2,i,j) * dgdsigma(i,j);
		}
	}
	Rfad[3] = dgamma * t + sigma(1,2) - sigmaVal(1,2);

	t = 0;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			t = t + Celastic(0,2,i,j) * dgdsigma(i,j);
		}
	}
	Rfad[4] = dgamma * t + sigma(0,2) - sigmaVal(0,2);

	t = 0;
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			t = t + Celastic(0,1,i,j) * dgdsigma(i,j);
		}
	}
	Rfad[5]  = dgamma * t + sigma(0,1) - sigmaVal(0,1);

	Rfad[6]  = dgamma * halpha(0,0) - alpha(0,0) + alphaVal(0,0);

	Rfad[7]  = dgamma * halpha(1,1) - alpha(1,1) + alphaVal(1,1);

	Rfad[8]  = dgamma * halpha(1,2) - alpha(1,2) + alphaVal(1,2);

	Rfad[9]  = dgamma * halpha(0,2) - alpha(0,2) + alphaVal(0,2);

	Rfad[10] = dgamma * halpha(0,1) - alpha(0,1) + alphaVal(0,1);

	if(kappa_flag == false)	Rfad[11] = dgamma * hkappa - kappa + kappaVal;
	else Rfad[11] = 0;

	// debugging
//	if(kappa_flag == false)Rfad[11] = -dgamma * hkappa - kappa + kappaVal;
//	else Rfad[11] = 0;

	Rfad[12] = f;


	// get ScalarT Residual
	for (int i=0; i<13; i++)
		R[i] = Rfad[i].val();

    //std::cout << "in assemble_Resid, R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;

	// get Jacobian
	for (int i=0; i<13; i++)
		for (int j=0; j<13; j++)
			dRdX[i + 13*j] = Rfad[i].dx(j);

	if(kappa_flag == true){
		for (int j=0; j<13; j++)
			dRdX[11 + 13*j] = 0.0;

		dRdX[11+13*11] = 1.0;
	}

}


} // end LCM
