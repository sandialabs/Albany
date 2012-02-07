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
CapModelStress<EvalT, Traits>::
CapModelStress(const Teuchos::ParameterList& p) :
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
  A          	   (p.get<double>("A Name")),
  B          	   (p.get<double>("B Name")),
  C          	   (p.get<double>("C Name")),
  theta        	   (p.get<double>("Theta Name")),
  R          	   (p.get<double>("R Name")),
  kappa0       	   (p.get<double>("Kappa0 Name")),
  W          	   (p.get<double>("W Name")),
  D1          	   (p.get<double>("D1 Name")),
  D2          	   (p.get<double>("D2 Name")),
  calpha       	   (p.get<double>("Calpha Name")),
  psi          	   (p.get<double>("Psi Name")),
  N          	   (p.get<double>("N Name")),
  L          	   (p.get<double>("L Name")),
  phi          	   (p.get<double>("Phi Name")),
  Q          	   (p.get<double>("Q Name"))
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
void CapModelStress<EvalT, Traits>::
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
void CapModelStress<EvalT, Traits>::
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
  	  LCM::Tensor<ScalarT> alphaVal = LCM::identity<ScalarT>();
  	  LCM::Tensor<ScalarT> sigmaN, strainN; // previous state

  	  for (std::size_t i = 0; i < numDims; ++i){
  	  	  for (std::size_t j = 0; j < numDims; ++j){
  	  		  sigmaN(i,j) = stressold(cell,qp,i,j);
  	  		  strainN(i,j) = strainold(cell,qp,i,j);
  	  		  sigmaVal(i,j) = sigmaVal(i,j) + stressold(cell,qp,i,j);
  	  		  alphaVal(i,j) = backStressold(cell,qp,i,j);
  	  	  }
  	  }

  	  ScalarT kappaVal = capParameterold(cell,qp);

  	  // make sure the cap starts from kappa0, and monotonically hardening (decreasing, more negative)
  	  if(kappaVal > kappa0){
  		  kappaVal = ScalarT(kappa0);
  		  //std::cout << "kappaVal > kappa0, make kappaVal = kappa0" << std::endl;
  	  }

  	  // check yielding
  	  ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);

  	  // plastic correction
  	  ScalarT dgamma = 0.0;
  	  if(f > 1.0e-10)
  	  {
  		LCM::Tensor<ScalarT> dfdsigma = compute_dfdsigma(sigmaN, alphaVal, kappaVal);

  		LCM::Tensor<ScalarT> dgdsigma = compute_dgdsigma(sigmaN, alphaVal, kappaVal);

  		LCM::Tensor<ScalarT> dfdalpha = - dfdsigma;

  		ScalarT dfdkappa = compute_dfdkappa(sigmaN, alphaVal, kappaVal);

		ScalarT J2_alpha = 0.5 * LCM::dotdot(alphaVal, alphaVal);

		LCM::Tensor<ScalarT> halpha = compute_halpha(dgdsigma, J2_alpha);

		ScalarT I1_dgdsigma = LCM::trace(dgdsigma);

		ScalarT dedkappa = compute_dedkappa(kappaVal);

		ScalarT hkappa;
		if(dedkappa != 0) hkappa = I1_dgdsigma / dedkappa;
		else hkappa = 0;

		ScalarT kai(0.0);
		kai = LCM::dotdot(dfdsigma, LCM::dotdot(Celastic, dgdsigma))
				- LCM::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

		LCM::Tensor<ScalarT> dfdotCe = LCM::dotdot(dfdsigma, Celastic);

		if(kai != 0)
			dgamma = LCM::dotdot(dfdotCe, depsilon) / kai;
		else
			dgamma = 0;

		// update
		sigmaVal -= dgamma * LCM::dotdot(Celastic, dgdsigma);

		alphaVal += dgamma * halpha;

		// restrictions on kappa, only allow monotonic decreasing (cap hardening)
		ScalarT dkappa = dgamma * hkappa;
		if (dkappa > 0) dkappa = 0;

		kappaVal += dkappa;

		// stress correction algorithm to avoid drifting from yield surface
		bool condition = false;
		int iteration = 0;
		while(condition == false){
			f = compute_f(sigmaVal, alphaVal, kappaVal);

			LCM::Tensor<ScalarT> dfdsigma = compute_dfdsigma(sigmaVal, alphaVal, kappaVal);

			LCM::Tensor<ScalarT> dgdsigma = compute_dgdsigma(sigmaVal, alphaVal, kappaVal);

			LCM::Tensor<ScalarT> dfdalpha = - dfdsigma;

			ScalarT dfdkappa = compute_dfdkappa(sigmaVal, alphaVal, kappaVal);

			J2_alpha = 0.5 * LCM::dotdot(alphaVal, alphaVal);

			halpha = compute_halpha(dgdsigma, J2_alpha);

			I1_dgdsigma = LCM::trace(dgdsigma);

			dedkappa = compute_dedkappa(kappaVal);

			if(dedkappa != 0)
				hkappa = I1_dgdsigma / dedkappa;
			else hkappa = 0;

			kai = LCM::dotdot(dfdsigma, LCM::dotdot(Celastic, dgdsigma));
			kai = kai - LCM::dotdot(dfdalpha, halpha) - dfdkappa * hkappa;

			if (std::abs(f) < 1.0e-10 )break;
			if (iteration > 20) {
			  	  // output for debug
			  	  std::cout<< "no stress correction after iteration = " << iteration
			  			 << " yield function abs(f) = " << abs(f) << std::endl;
				break;
			}

			ScalarT delta_gamma;
			if (kai != 0)
				delta_gamma = f / kai;
			else
				delta_gamma = 0;

			LCM::Tensor<ScalarT> sigmaK, alphaK;
			ScalarT kappaK;

			// restrictions on kappa, only allow monotonic decreasing
			dkappa = delta_gamma * hkappa;
			if (dkappa > 0) dkappa = 0;

			sigmaK = sigmaVal - delta_gamma * LCM::dotdot(Celastic, dgdsigma);
			alphaK = alphaVal + delta_gamma * halpha;
			kappaK = kappaVal + dkappa;

			ScalarT fpre = compute_f(sigmaK, alphaK, kappaK);

			if (std::abs(fpre) > std::abs(f)){
				// if the corrected stress is further away from yield surface, then use normal correction
				ScalarT dfdotdf = LCM::dotdot(dfdsigma, dfdsigma);
				if (dfdotdf != 0)
					delta_gamma = f / dfdotdf;
				else
					delta_gamma = 0.0;

				sigmaK = sigmaVal - delta_gamma * dfdsigma;
				alphaK = alphaVal;
				kappaK = kappaVal;
			}

			sigmaVal = sigmaK;
			alphaVal = alphaK;
			kappaVal = kappaK;

			iteration++;

		}// end of stress correction

  	  }// end of plastic correction

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
typename CapModelStress<EvalT, Traits>::ScalarT
CapModelStress<EvalT, Traits>::compute_f(LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
{

	LCM::Tensor<ScalarT> xi = sigma - alpha;

	ScalarT I1 = LCM::trace(xi), p = I1 / 3;

	LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>();

	ScalarT J2 = 0.5 * LCM::dotdot(s, s);

	ScalarT J3 = LCM::det(s);

	ScalarT Gamma = 1.0;
	if (psi!=0 && J2!=0) Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
			+ (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

	ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

	ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

	ScalarT X = kappa -  R * Ff_kappa;

	ScalarT Fc = 1.0;

	if((kappa - I1) > 0 && ((X - kappa) != 0))
		Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

	return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
}

template<typename EvalT, typename Traits>
LCM::Tensor<typename CapModelStress<EvalT, Traits>::ScalarT>
CapModelStress<EvalT, Traits>::compute_dfdsigma(LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
{
	LCM::Tensor<ScalarT> dfdsigma;

	LCM::Tensor<ScalarT> xi = sigma - alpha;

	ScalarT I1 = LCM::trace(xi), p = I1 / 3;

	LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>();

	ScalarT J2 = 0.5 * LCM::dotdot(s, s);

	ScalarT J3 = LCM::det(s);

	LCM::Tensor<ScalarT> id = LCM::identity<ScalarT>();
	LCM::Tensor<ScalarT> dI1dsigma = id;
	LCM::Tensor<ScalarT> dJ2dsigma = s;
	LCM::Tensor<ScalarT> dJ3dsigma;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			dJ3dsigma(i,j) = s(i,j) * s(i,j) - 2 * J2 * id(i,j) / 3;

	ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

	ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

	ScalarT X = kappa - R * Ff_kappa;

	ScalarT Fc = 1.0;

	if((kappa - I1) > 0)
		Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

	ScalarT Gamma = 1.0;
	if (psi!=0 && J2!=0) Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
			+ (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

	// derivatives
	ScalarT dFfdI1 = -(B * C * std::exp(B * I1) + theta);

	ScalarT dFcdI1 = 0.0;
	if((kappa - I1) > 0 && ((X - kappa) != 0))
		dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

	ScalarT dfdI1 = -(Ff_I1 - N) * ( 2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

	ScalarT dGammadJ2 = 0.0;
	if (J2 != 0)
		dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);

	ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

	ScalarT dGammadJ3 = 0.0;
	if(J2 != 0)
		dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);

	ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

	dfdsigma = dfdI1 * dI1dsigma + dfdJ2 * dJ2dsigma + dfdJ3 * dJ3dsigma;

	return dfdsigma;
}

template<typename EvalT, typename Traits>
LCM::Tensor<typename CapModelStress<EvalT, Traits>::ScalarT>
CapModelStress<EvalT, Traits>::compute_dgdsigma(LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
{
	LCM::Tensor<ScalarT> dgdsigma;

	LCM::Tensor<ScalarT> xi = sigma - alpha;

	ScalarT I1 = LCM::trace(xi), p = I1 / 3;

	LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>();

	ScalarT J2 = 0.5 * LCM::dotdot(s, s);

	ScalarT J3 = LCM::det(s);

	LCM::Tensor<ScalarT> id = LCM::identity<ScalarT>();
	LCM::Tensor<ScalarT> dI1dsigma = id;
	LCM::Tensor<ScalarT> dJ2dsigma = s;
	LCM::Tensor<ScalarT> dJ3dsigma;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			dJ3dsigma(i,j) = s(i,j) * s(i,j) - 2 * J2 * id(i,j) / 3;

	ScalarT Ff_I1 = A - C * std::exp(L * I1) - phi * I1;

	ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

	ScalarT X = kappa - Q * Ff_kappa;

	ScalarT Fc = 1.0;

	if((kappa - I1) > 0)
		Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);

	ScalarT Gamma = 1.0;
	if (psi!=0 && J2!=0) Gamma = 0.5 * (1 - 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)
			+ (1 + 3.0 * std::sqrt(3.0) * J3 / 2 / std::pow(J2, 1.5)) / psi);

	// derivatives
	ScalarT dFfdI1 = -(L * C * std::exp(L * I1) + phi);

	ScalarT dFcdI1 = 0.0;
	if((kappa - I1) > 0 && ((X - kappa) != 0))
		dFcdI1 = -2.0 * (I1 - kappa) / (X - kappa) / (X - kappa);

	ScalarT dfdI1 = -(Ff_I1 - N) * ( 2 * Fc * dFfdI1 + (Ff_I1 - N) * dFcdI1);

	ScalarT dGammadJ2 = 0.0;
	if (J2 != 0)
		dGammadJ2 = 9.0 * std::sqrt(3.0) * J3 / std::pow(J2, 2.5) / 8.0 * (1.0 - 1.0 / psi);

	ScalarT dfdJ2 = 2.0 * Gamma * dGammadJ2 * J2 + Gamma * Gamma;

	ScalarT dGammadJ3 = 0.0;
	if(J2 != 0)
		dGammadJ3 = -3.0 * std::sqrt(3.0) / std::pow(J2, 1.5) / 4.0 * (1.0 - 1.0 / psi);

	ScalarT dfdJ3 = 2.0 * Gamma * dGammadJ3 * J2;

	dgdsigma = dfdI1 * dI1dsigma + dfdJ2 * dJ2dsigma + dfdJ3 * dJ3dsigma;

	return dgdsigma;
}

template<typename EvalT, typename Traits>
typename CapModelStress<EvalT, Traits>::ScalarT
CapModelStress<EvalT, Traits>::compute_dfdkappa(LCM::Tensor<ScalarT> & sigma, LCM::Tensor<ScalarT> & alpha, ScalarT & kappa)
{
	ScalarT dfdkappa;
	LCM::Tensor<ScalarT> dfdsigma;

	LCM::Tensor<ScalarT> xi = sigma - alpha;

	ScalarT I1 = LCM::trace(xi), p = I1 / 3;

	LCM::Tensor<ScalarT> s = xi - p * LCM::identity<ScalarT>();

	ScalarT J2 = 0.5 * LCM::dotdot(s, s);

	ScalarT J3 = LCM::det(s);

	ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;

	ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;

	ScalarT X = kappa - R * Ff_kappa;

	ScalarT dFcdkappa = 0.0;

	if((kappa - I1) > 0 && ((X - kappa) != 0))
		dFcdkappa = 2 * (I1 - kappa) *
		((X - kappa) + R*(I1-kappa)*(theta+B*C*std::exp(B*kappa)))
		/ (X-kappa) / (X-kappa) / (X-kappa);

	dfdkappa = -dFcdkappa * (Ff_I1 - N) * (Ff_I1 - N);

	return dfdkappa;
}
template<typename EvalT, typename Traits>
typename CapModelStress<EvalT, Traits>::ScalarT
CapModelStress<EvalT, Traits>::compute_Galpha(ScalarT & J2_alpha)
{
	if(N != 0)
		return 1.0 - std::pow(J2_alpha, 0.5) / N;
	else
		return 0.0;
}

template<typename EvalT, typename Traits>
LCM::Tensor<typename CapModelStress<EvalT, Traits>::ScalarT>
CapModelStress<EvalT, Traits>::compute_halpha(LCM::Tensor<ScalarT> & dgdsigma, ScalarT & J2_alpha)
{

	ScalarT Galpha = compute_Galpha(J2_alpha);

	ScalarT I1 = LCM::trace(dgdsigma), p = I1 / 3;

	LCM::Tensor<ScalarT> s = dgdsigma - p * LCM::identity<ScalarT>();

	LCM::Tensor<ScalarT> halpha;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			halpha(i,j) = calpha * Galpha * s(i,j);
		}
	}

	return halpha;
}

template<typename EvalT, typename Traits>
typename CapModelStress<EvalT, Traits>::ScalarT
CapModelStress<EvalT, Traits>::compute_dedkappa(ScalarT & kappa)
{
	ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;

	ScalarT X0 = kappa0 -  Q * Ff_kappa0;

	ScalarT Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;

	ScalarT X = kappa -  Q * Ff_kappa;

	ScalarT dedX = (D1 - 2 * D2 * (X-X0)) * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;

	ScalarT dXdkappa = 1 + Q * C * L * std::exp(L * kappa) + Q * phi;

	return dedX * dXdkappa;
}

//**********************************************************************
} // end LCM
