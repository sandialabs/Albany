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
GursonSDStress<EvalT, Traits>::
GursonSDStress(const Teuchos::ParameterList& p) :
  elasticModulus   (p.get<std::string>                   ("Elastic Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatio    (p.get<std::string>                   ("Poissons Ratio Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  strain           (p.get<std::string>                   ("Strain Name"),
   	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  voidVolume       (p.get<std::string>                   ("Void Volume Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  yieldStrength	   (p.get<std::string>                   ("Yield Strength Name"),
				p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  f0          	   (p.get<double>("f0 Name")),
  sigmaY           (p.get<double>("sigmaY Name")),
  kw          	   (p.get<double>("kw Name")),
  N        	   	   (p.get<double>("N Name")),
  q1          	   (p.get<double>("q1 Name")),
  q2       	   	   (p.get<double>("q2 Name")),
  q3          	   (p.get<double>("q3 Name")),
  eN          	   (p.get<double>("eN Name")),
  sN          	   (p.get<double>("sN Name")),
  fN       	   	   (p.get<double>("fN Name")),
  fc          	   (p.get<double>("fc Name")),
  ff          	   (p.get<double>("ff Name"))
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
    voidVolumeName = p.get<std::string>("Void Volume Name")+"_old";
    yieldStrengthName = p.get<std::string>("Yield Strength Name")+"_old";

    // evaluated fields
    this->addEvaluatedField(stress);
    this->addEvaluatedField(voidVolume);
    this->addEvaluatedField(yieldStrength);

    this->setName("Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void GursonSDStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(elasticModulus,fm);
  if (numDims>1) this->utils.setFieldData(poissonsRatio,fm);
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(strain,fm);
  this->utils.setFieldData(voidVolume,fm);
  this->utils.setFieldData(yieldStrength,fm);
}

//**********************************************************************

template<typename EvalT, typename Traits>
void GursonSDStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
	  // previous state
	  Albany::MDArray strainold = (*workset.stateArrayPtr)[strainName];
	  Albany::MDArray stressold = (*workset.stateArrayPtr)[stressName];
	  Albany::MDArray voidVolumeold = (*workset.stateArrayPtr)[voidVolumeName];
	  Albany::MDArray yieldStrengthold = (*workset.stateArrayPtr)[yieldStrengthName];

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
		LCM::Tensor<ScalarT> depsilon(0.0),sigmaN(0.0);
		for (std::size_t i = 0; i < numDims; ++i){
		  for (std::size_t j = 0; j < numDims; ++j){
			  depsilon(i,j) = strain(cell,qp,i,j) - strainold(cell,qp,i,j);
			  sigmaN(i,j) = stressold(cell,qp,i,j);
		  }
		}

		// trial state
		// to correctly initialize fvoid and sigmaM to nonzero values?
		// use voidVolumeold and yieldStrengthold to store incremental values
		LCM::Tensor<ScalarT> sigmaVal = sigmaN + LCM::dotdot(Celastic, depsilon);

		ScalarT fvoid_Delta = voidVolumeold(cell,qp);
		ScalarT sigmaM_Delta = yieldStrengthold(cell,qp);

		ScalarT fvoidN = f0 + fvoid_Delta;
		ScalarT sigmaMN = sigmaY + sigmaM_Delta;

		ScalarT fvoidVal  = fvoidN;
		ScalarT sigmaMVal = sigmaMN;

		//std::cout << "fvoidN = " << fvoidN << std::endl;
		//std::cout << "sigmaMN =" << sigmaMN << std::endl;

		// check yielding
		ScalarT FVal = compute_F(sigmaVal, fvoidVal, sigmaMVal);
		//std::cout << "stress old = " << sigmaN << std::endl;

		//std::cout << "depsilon = " << depsilon << std::endl;
		//std::cout << "Ftrial = " << FVal << std::endl;


		ScalarT dgamma = 0.0;
		if(FVal > 1.0e-10){
			LCM::Tensor<DFadType> sigmaD(0.0);
			DFadType fvoidD, sigmaMD;
			// convert to DFadType, use previous state, not trial state!
			getDFadType(sigmaN, fvoidN, sigmaMN, sigmaD, fvoidD, sigmaMD);

			DFadType F = compute_F(sigmaD, fvoidD, sigmaMD);

			LCM::Tensor<ScalarT> dFdsigma = compute_dFdsigma(F);

			ScalarT dFdfvoid = F.dx(6);

			ScalarT dFdsigmaM = F.dx(7);

			ScalarT hsigmaM = compute_hsigmaM(sigmaN, fvoidN, sigmaMN, dFdsigma, mu);

			ScalarT hfvoid = compute_hfvoid(sigmaN, fvoidN, sigmaMN, dFdsigma, mu);

			ScalarT kai(0.0);
			kai = LCM::dotdot(dFdsigma, LCM::dotdot(Celastic, dFdsigma));
			kai = kai - dFdfvoid * hfvoid - dFdsigmaM * hsigmaM;

			LCM::Tensor<ScalarT> dFdotCe = LCM::dotdot(dFdsigma, Celastic);

			// incremental consistency parameter
			if(kai != 0)
				dgamma = LCM::dotdot(dFdotCe, depsilon) / kai;
			else
				dgamma = 0;

			//update
			sigmaVal -= dgamma * LCM::dotdot(Celastic, dFdsigma);
			fvoidVal += dgamma * hfvoid;
			sigmaMVal += dgamma * hsigmaM;

			// stress correction to prevent drifting
			bool condition = false; int iteration = 0;
			while(condition ==  false){
				getDFadType(sigmaVal, fvoidVal, sigmaMVal, sigmaD, fvoidD, sigmaMD);

				F = compute_F(sigmaD, fvoidD, sigmaMD);

				if (std::abs(F) < 1.0e-10) break;
				if (iteration > 15){
					//std::cout << "no convergence in stress correction after " << iteration << " iterations" << std::endl;
					//std::cout << "F= " << F.val() << std::endl;
					break;
				}

				dFdsigma = compute_dFdsigma(F);

				dFdfvoid = F.dx(6);

				dFdsigmaM = F.dx(7);

				hsigmaM = compute_hsigmaM(sigmaN, fvoidN, sigmaMN, dFdsigma, mu);

				hfvoid = compute_hfvoid(sigmaN, fvoidN, sigmaMN, dFdsigma, mu);

				kai = LCM::dotdot(dFdsigma, LCM::dotdot(Celastic, dFdsigma));
				kai = kai - dFdfvoid * hfvoid - dFdsigmaM * hsigmaM;

				ScalarT delta_gamma;
				if(kai!=0) delta_gamma = F.val() / kai;
				else delta_gamma = 0;

				LCM::Tensor<ScalarT> sigmaK(0.0);
				ScalarT fvoidK, sigmaMK;

				sigmaK = sigmaVal - delta_gamma * LCM::dotdot(Celastic, dFdsigma);
				fvoidK = fvoidVal + delta_gamma * hfvoid;
				sigmaMK = sigmaMVal + delta_gamma * hsigmaM;

				ScalarT Fpre = compute_F(sigmaK, fvoidK, sigmaMK);

				if(std::abs(Fpre) > std::abs(F)){
					// if the corrected stress is further away from yield surface, then use normal correction
					ScalarT dFdotdF = LCM::dotdot(dFdsigma, dFdsigma);
					if(dFdotdF != 0) delta_gamma = F.val() / dFdotdF;
					else delta_gamma = 0.0;

					sigmaK = sigmaVal - delta_gamma * dFdsigma;
					fvoidK = fvoidVal;
					sigmaMK = sigmaMVal;
				}

				sigmaVal = sigmaK;
				fvoidVal = fvoidK;
				sigmaMVal = sigmaMK;

				iteration ++;

			} // end of stress correction
		}// end of plastic correction

	  // update
  	  for (std::size_t i = 0; i < numDims; ++i)
  	  	  for (std::size_t j = 0; j < numDims; ++j)
  	  		  stress(cell,qp,i,j) = sigmaVal(i,j);

  	  voidVolume(cell,qp) = fvoidVal - f0;
  	  yieldStrength(cell,qp) = sigmaMVal - sigmaY;

  	  //voidVolume(cell,qp) = fvoidVal;
  	  //yieldStrength(cell,qp) = sigmaMVal;

	} //loop over qps

  }//loop over cell


} // end of evaluateFields


//**********************************************************************
// all local functions
template<typename EvalT, typename Traits>
typename EvalT::ScalarT
GursonSDStress<EvalT, Traits>::compute_F(LCM::Tensor<ScalarT> & sigmaVal, ScalarT & fvoidVal, ScalarT & sigmaMVal)
{
	ScalarT p = LCM::trace(sigmaVal);
	p = p / 3;

	LCM::Tensor<ScalarT> s = sigmaVal - p * LCM::identity<ScalarT>();

	ScalarT q = LCM::dotdot(s,s);

	q = std::sqrt(q * 3 / 2);

	ScalarT pt = 0;
	if(sigmaMVal != 0)
		pt = 3 * q2 * p / 2 / sigmaMVal;

	ScalarT fstar;
	fstar = fvoidVal;

	// not yet enabled for fstar approaching fc and ff
//	ScalarT ffbar = (q1 + std::sqrt(q1 * q1 -  q3)) / q3;

//	if(fvoidVal <= fc)
//		fstar = fvoidVal;
//	else if((fvoidVal > fc) && (fvoidVal < ff))
//		fstar = fc + (ffbar - fc) * (fvoidVal - fc) / (ff - fc);
//	else
//		fstar = ffbar;

	ScalarT F(0.0);

	if (sigmaMVal != 0)
		F = (q/sigmaMVal) * (q/sigmaMVal)
			+ 2 * q1 * fstar * std::cosh(pt) - (1 + q3 * fstar * fstar);
	return F;
}

template<typename EvalT, typename Traits>
void
GursonSDStress<EvalT, Traits>::getDFadType(LCM::Tensor<ScalarT> & sigmaVal, ScalarT & fvoidVal,ScalarT & sigmaMVal,
		LCM::Tensor<DFadType> & sigmaD, DFadType & fvoidD, DFadType & sigmaMD)
{
	//
	sigmaD(0,0)  = DFadType(8, 0, sigmaVal(0,0));
	sigmaD(1,1)  = DFadType(8, 1, sigmaVal(1,1));
	sigmaD(2,2)  = DFadType(8, 2, sigmaVal(2,2));
	sigmaD(1,2)  = DFadType(8, 3, sigmaVal(1,2));  sigmaD(2,1) = sigmaD(1,2);
	sigmaD(0,2)  = DFadType(8, 4, sigmaVal(0,2));  sigmaD(2,0) = sigmaD(0,2);
	sigmaD(0,1)  = DFadType(8, 5, sigmaVal(0,1));  sigmaD(1,0) = sigmaD(0,1);
	fvoidD		 = DFadType(8, 6, fvoidVal);
	sigmaMD		 = DFadType(8, 7, sigmaMVal);

	return;
}

template<typename EvalT, typename Traits>
typename GursonSDStress<EvalT, Traits>::DFadType
GursonSDStress<EvalT, Traits>::compute_F(LCM::Tensor<DFadType> & sigma,
		DFadType & fvoid, DFadType & sigmaM)
{
	DFadType p = LCM::trace(sigma);
	p = p / 3;

	LCM::Tensor<DFadType> s = sigma - p * LCM::identity<DFadType>();

	DFadType q = LCM::dotdot(s,s);

	q = std::sqrt(q * 3 / 2);

	DFadType pt = 0;
	if (sigmaM != 0)
		pt = 3 * q2 * p / 2 / sigmaM;

	DFadType fstar;
	fstar = fvoid;

	// not yet enabled for fstar approaching fc and ff
//	ScalarT ffbar = (q1 + std::sqrt(q1 * q1 -  q3)) / q3;
//	if(fvoid <= fc)
//		fstar = fvoid;
//	else if((fvoid > fc) && (fvoid < ff))
//		fstar = fc + (ffbar - fc) * (fvoid - fc) / (ff - fc);
//	else
//		fstar = ffbar;

	DFadType F(0.0);
	if(sigmaM != 0)
	{
		if(q != 0)
			F = (q/sigmaM) * (q/sigmaM)
							+ 2 * q1 * fstar * std::cosh(pt) - (1 + q3 * fstar * fstar);
		else
			F = 2 * q1 * fstar * std::cosh(pt) - (1 + q3 * fstar * fstar);
	}

	return F;
}

template<typename EvalT, typename Traits>
LCM::Tensor<typename EvalT::ScalarT>
GursonSDStress<EvalT, Traits>::compute_dFdsigma(DFadType & F)
{
	LCM::Tensor<ScalarT> dFdsigma(0.0);

	dFdsigma(0,0) = F.dx(0); dFdsigma(0,1) = F.dx(5); dFdsigma(0,2) = F.dx(4);
	dFdsigma(1,0) = F.dx(5); dFdsigma(1,1) = F.dx(1); dFdsigma(1,2) = F.dx(3);
	dFdsigma(2,0) = F.dx(4); dFdsigma(2,1) = F.dx(3); dFdsigma(2,2) = F.dx(2);

	return dFdsigma;
}

template<typename EvalT, typename Traits>
typename EvalT::ScalarT
GursonSDStress<EvalT, Traits>::compute_hsigmaM(LCM::Tensor<ScalarT> & sigma, ScalarT & fvoid, ScalarT & sigmaM,
		LCM::Tensor<ScalarT> & dFdsigma, ScalarT & mu)
{

	ScalarT Nt = (1 - N) / N;

	ScalarT ratio = sigmaM / sigmaY;

	ScalarT hsigmaM = 0;

	ScalarT hM = 0;

//	if(sigmaM >= sigmaY)
//		hM = E / (std::pow(ratio, Nt) / N - 1);

	// neglect elasticity
//	if(sigmaM >= sigmaY)
//		hM = E / (std::pow(ratio, Nt) / N );

	//std::cout<< "hm as in Aravas 1987" << std::endl;
	if(sigmaM >= sigmaY)
		hM = 3 * mu / (std::pow(ratio, Nt) / N - 1);

	if(sigmaM != 0)
		hsigmaM = hM * LCM::dotdot(sigma, dFdsigma) / (1 - fvoid) / sigmaM;

	return hsigmaM;
}

template<typename EvalT, typename Traits>
typename EvalT::ScalarT
GursonSDStress<EvalT, Traits>::compute_hfvoid(LCM::Tensor<ScalarT> & sigma,	ScalarT & fvoid, ScalarT & sigmaM,
		LCM::Tensor<ScalarT> & dFdsigma, ScalarT & mu)
{
	ScalarT p = LCM::trace(sigma);
	p = p / 3;

	LCM::Tensor<ScalarT> s = sigma - p * LCM::identity<ScalarT>();

	ScalarT q = LCM::dotdot(s,s);
	q = std::sqrt(q * 3 / 2);

	ScalarT J3 = LCM::det(s);

	ScalarT omega = 0;

	if (q != 0)
		omega = 1 - (27.0 * J3 / (2 * q * q * q)) * (27.0 * J3 / (2 * q * q * q));

	// void growth
	ScalarT hfvoid_g;

	if (q !=0 )
		hfvoid_g = (1 - fvoid) * LCM::trace(dFdsigma)
					+ kw * fvoid * omega * LCM::dotdot(s,dFdsigma) / q;
	else
		hfvoid_g = (1 - fvoid) * LCM::trace(dFdsigma);

	// void nucleation
	ScalarT A, epsMp(0.0);

	ScalarT Nt = 1 / N;

	ScalarT ratio = sigmaM / sigmaY;

//	if (sigmaM >= sigmaY)
//		epsMp = (sigmaY / E) * std::pow(ratio, Nt) - sigmaM / E;

	// neglect elasticity
//	if (sigmaM >= sigmaY)
//		epsMp = (sigmaY / E) * std::pow(ratio, Nt);

	//std::cout<< "epsMp as in Aravas 1987" << std::endl;
	if (sigmaM >= sigmaY)
		epsMp = (sigmaY / mu /3) * std::pow(ratio, Nt) - sigmaM / mu / 3;

	const double pi = acos(-1.0);
	ScalarT eratio = -0.5 * (epsMp - eN) * (epsMp - eN) / sN /sN;

	if (p >= 0)
		A = fN / sN / (std::sqrt(2.0 * pi)) * std::exp(eratio);
	else
		A = 0;

	ScalarT hfvoid_n = 0;
	if (sigmaM != 0 )
		hfvoid_n = A * LCM::dotdot(sigma, dFdsigma) / (1 - fvoid) / sigmaM;

	// with both void growth and nucleation
	ScalarT hfvoid = hfvoid_g + hfvoid_n;

	return hfvoid;
}
//**********************************************************************
} // end LCM
