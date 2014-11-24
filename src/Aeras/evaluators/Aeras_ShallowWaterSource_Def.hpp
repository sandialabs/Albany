//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

const double pi = 3.1415926535897932385;
 
//**********************************************************************
template<typename EvalT, typename Traits>
ShallowWaterSource<EvalT, Traits>::
ShallowWaterSource(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  gravity (Aeras::ShallowWaterConstants::self().gravity),
  Omega(2.0*(Aeras::ShallowWaterConstants::self().pi)/(24.*3600.)),
  //OG should source be node vector? like residual?
  source    (p.get<std::string> ("Shallow Water Source QP Variable Name"), dl->qp_vector)
{
  Teuchos::ParameterList* shallowWaterList =
   p.get<Teuchos::ParameterList*>("Parameter List");

   std::string sourceTypeString = shallowWaterList->get<std::string>("SourceType", "None");

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  const bool invalidString = (sourceTypeString != "None" && sourceTypeString != "TC4");
  TEUCHOS_TEST_FOR_EXCEPTION( invalidString,
                                  std::logic_error,
                                  "Unknown shallow water source string of " << sourceTypeString
                                          << " encountered. " << std::endl);

  if (sourceTypeString == "None"){
    sourceType = NONE;
  }
  else if (sourceTypeString == "TC4") {
   *out << "Setting TC4 source.  To be implemented by Oksana." << std::endl;
   sourceType = TC4;

  }

  this->addDependentField(sphere_coord);
  this->addEvaluatedField(source);

  std::vector<PHX::DataLayout::size_type> dims;
  
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->qp_vector->dimensions(dims);
  vecDim  = dims[2]; //# of dofs/node

  this->setName("ShallowWaterSource"+PHX::TypeString<EvalT>::value);
  
  myPi = Aeras::ShallowWaterConstants::self().pi;
  
  earthRadius = Aeras::ShallowWaterConstants::self().earthRadius;
  
  
  //////////parameters for SW TC4 source
  SU0 = 20.;
  PHI0 = 1.0e5;
  
  RLON0 = 0.;
  RLAT0 = myPi/4.;
  NPWR = 14.;
  
  ALFA =  -0.03*(PHI0/(2.*Omega*sin(myPi/4.)));
  SIGMA = (2.*earthRadius/1.0e6)*(2.*earthRadius/1.0e6);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(source,fm);
  this->utils.setFieldData(sphere_coord,fm);
}

//**********************************************************************
//A concrete (non-virtual) implementation of getValue is needed for code to compile. 
//Do we really need it though for this problem...?
template<typename EvalT,typename Traits>
typename ShallowWaterSource<EvalT,Traits>::ScalarT& 
ShallowWaterSource<EvalT,Traits>::getValue(const std::string &n)
{
  static ScalarT junk(0);
  return junk;
}


//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if (sourceType == NONE) {
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        for (std::size_t i = 0; i < vecDim; ++i) { //loop over # of dofs/node
          source(cell, qp, i) = 0;
        }
      }
    }
  }
  else if(sourceType == TC4) {
    
    ScalarT A = earthRadius;
  
    const RealType time = workset.current_time; //current time from workset
    
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        //for (std::size_t i = 0; i < vecDim; ++i) { //loop over # of dofs/node
          const MeshScalarT theta = sphere_coord(cell, qp, 0);
          const MeshScalarT lambda = sphere_coord(cell, qp, 1);

          //source(cell, qp, i) = 0.0; //set this to some function involving time
          
          ScalarT TMSHFT = SU0*time/A;
          ScalarT DFDM   = 2.0*Omega;
          ScalarT AI    = 1.0/A;
          ScalarT A2I   = 1.0/(A*A);
    
    //DO 510 J=1,NLAT
    //RLAT = GLAT(J)

          
          ScalarT RLAT = theta;
          
          ScalarT SNJ  = sin(RLAT);
          ScalarT CSJ  = cos(RLAT)*cos(RLAT);

          ScalarT BUB    = bubfnc(RLAT)*cos(RLAT);
          ScalarT DBUB   = dbubf(RLAT);
          ScalarT D2BUB  = d2bubf(RLAT);

          ScalarT COR    = 2.0*Omega*SNJ;

          ScalarT SRCSJ  = cos(RLAT);
          ScalarT TMPRY  = tan(RLAT); //////potential problem at poles

          ScalarT CSJI   = 1.0/CSJ;
          ScalarT CSJ2I  = 1.0/(CSJ*CSJ);
          ScalarT ACSJI  = 1.0/(A*CSJ);
          ScalarT AACSJI = 1.0/(A*A*CSJ);
          ScalarT ACSJ2I = 1.0/((A*CSJ)*(A*CSJ));
    
        ScalarT RLON = lambda;
    //DO 100 I=1,NLON
    //RLON = GLON 
    //           CALCULATE C AND ALL OF ITS DERIVATIVES FOR THIS TIME STEP

          ScalarT C = sin(RLAT0)*SNJ + cos(RLAT0)*SRCSJ*cos(RLON-TMSHFT-RLON0);
  
          ScalarT DCDM    = sin(RLAT0) - cos(RLON-TMSHFT-RLON0)*cos(RLAT0)*TMPRY;
          ScalarT DCDL    = -cos(RLAT0)*SRCSJ*sin(RLON-TMSHFT-RLON0);
          ScalarT D2CDM   = -cos(RLAT0)*cos(RLON-TMSHFT-RLON0)*CSJI/SRCSJ;

          ScalarT D2CDL   = -cos(RLAT0)*SRCSJ*cos(RLON-TMSHFT-RLON0);

          ScalarT D3CDM   = D2CDM *3.0*SNJ*CSJI;

          ScalarT D3CDL   = -DCDL;

          ScalarT DMDCDL  = +cos(RLAT0)*sin(RLON-TMSHFT-RLON0)*TMPRY;

          ScalarT DLD2CM  = +cos(RLAT0)*sin(RLON-TMSHFT-RLON0)*CSJI/SRCSJ;

          ScalarT DMD2CL  = +cos(RLAT0)*cos(RLON-TMSHFT-RLON0)*TMPRY;

        //cond, potential singularity
          ScalarT PSIB    = ALFA*exp(-SIGMA*((1.0-C )/(1.0+C )));
        //cond, potential singularity
          ScalarT TMP1    = 2.0*SIGMA*PSIB /((1.0 + C )*(1.0 + C));
        
          ScalarT TMP2    = (SIGMA - (1.0 + C ))/((1.0 + C )*(1.0 + C));
          ScalarT TMP3    = (((1.0+C )*(1.0+C))-2.0*SIGMA*(1.0+C ))/
                           ((1.0 + C )*(1.0 + C)*(1.0 + C)*(1.0 + C));

          ScalarT DKDM    = TMP1 *DCDM;

          ScalarT DKDL    = TMP1 *DCDL;

          ScalarT D2KDM   = TMP1 *(D2CDM  + 2.0*(DCDM *DCDM)*TMP2 );

          ScalarT D2KDL   = TMP1 *(D2CDL  + 2.0*(DCDL *DCDL)*TMP2 );

          ScalarT D3KDM   = TMP1 *(D3CDM  + 2.0*(DCDM *DCDM*DCDM)*TMP3  + 2.0*DCDM *TMP2
                                  *(3.0*D2CDM  + 2.0*(DCDM *DCDM)*TMP2 ));
  
          ScalarT D3KDL   = TMP1 *(D3CDL  + 2.0*(DCDL *DCDL*DCDL)*TMP3  + 2.0*DCDL *TMP2
                                  *(3.0*D2CDL  + 2.0*(DCDL *DCDL)*TMP2 ));

          ScalarT DLDKDM  = TMP1 *(DMDCDL  + 2.0*DCDL *DCDM *TMP2 );

          ScalarT DMD2KL  = TMP1 *(DMD2CL  + 2.0*(DCDL *DCDL)
                                  *DCDM *TMP3  + 2.0*DCDM *TMP2
                         *(D2CDL  + 2.0*(DCDL *DCDL)*TMP2 )
                                  + 4.0*DCDL *DMDCDL *TMP2 );

          ScalarT DLD2KM  = TMP1 *(DLD2CM  + 2.0*(DCDM *DCDM)
                         *DCDL *TMP3  + 2.0*DCDL *TMP2 
                         *(D2CDM  + 2.0*(DCDM *DCDM)*TMP2 )
                                + 4.0*DCDM *DMDCDL *TMP2 );

          ScalarT UT     = BUB - CSJ*DKDM *AI;
          ScalarT VT     = DKDL *AI;
          ScalarT DUTDL  = -CSJ*DLDKDM *AI;
          ScalarT DVTDL  = D2KDL *AI;
          ScalarT DUTDM  = DBUB - (CSJ*D2KDM  - 2.0*SNJ*DKDM )*AI;
          ScalarT DVTDM  = DLDKDM *AI;

          TMP1  = (CSJ*SU0)/(A*A)*DLDKDM;
          TMP1  = TMP1  + UT *ACSJI*DUTDL;
          TMP1  = TMP1  + VT *AI*DUTDM;
    
          ScalarT ETAFCG = TMP1  + COR*(AI*DKDL -VT ); //U
    
          TMP3      = COR*DKDL;
          ScalarT PHIFCG = -SU0*AI*TMP3  + UT *TMP3 *ACSJI;
          PHIFCG = PHIFCG + AI*VT *(PSIB *DFDM+COR*DKDM );
          PHIFCG = PHIFCG - VT *BUB*CSJI*(COR + BUB*ACSJI*SNJ);  //H

          TMP3  = - SU0*A2I*D2KDL;
          TMP3  = TMP3  + UT *ACSJI*DVTDL;
          TMP3  = TMP3  + VT *AI*DVTDM;
          TMP3  = TMP3  + (UT *UT +VT *VT )*SNJ*ACSJI;
    
          ScalarT DIVFCG = TMP3  + CSJ*AI*(COR*DKDM +PSIB *DFDM);
          DIVFCG = DIVFCG + COR*(UT -BUB) - ACSJI*SNJ*BUB*BUB; //V
    
          
          source(cell, qp, 0) = PHIFCG;
          source(cell, qp, 1) = ETAFCG;
          source(cell, qp, 2) = DIVFCG;
        
      }
    }
    
  }
}
 
template<typename EvalT,typename Traits>
  typename ShallowWaterSource<EvalT,Traits>::ScalarT
 ShallowWaterSource<EvalT,Traits>::d2bubf(const ScalarT lat){
    ScalarT rmu = sin(lat);
    ScalarT coslat = cos(lat);
    return 8.0*SU0* std::pow((2.0*rmu*coslat), (NPWR-3.)) *rmu*
          ((NPWR-1.)*NPWR+rmu*rmu*((NPWR-1.)-
                                   (2.*NPWR*(2.*NPWR+1.))*coslat*coslat));
  }
  
  
template<typename EvalT,typename Traits>
    typename ShallowWaterSource<EvalT,Traits>::ScalarT
  ShallowWaterSource<EvalT,Traits>::dbubf(const ScalarT lat){
    ScalarT rmu = sin(lat);
    ScalarT coslat = cos(lat);
    return 2.*SU0*std::pow(2.*rmu*coslat,NPWR-1.)
    *(NPWR-(2.*NPWR+1.)*rmu*rmu);
  }

template<typename EvalT,typename Traits>
    typename ShallowWaterSource<EvalT,Traits>::ScalarT
  ShallowWaterSource<EvalT,Traits>::bubfnc(const ScalarT lat){
    return SU0*std::pow((2.*sin(lat)*cos(lat)), NPWR);
  }

  
template<typename EvalT,typename Traits>
void ShallowWaterSource<EvalT,Traits>::get_coriolis(std::size_t cell, Intrepid::FieldContainer<ScalarT>  & coriolis) {
    
    coriolis.initialize();
    double alpha = 0.0;//1.047;  /*must match what is in initial condition for TC2 and TC5.
    //see AAdatpt::AerasZonal analytic function. */
    
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      const MeshScalarT lambda = sphere_coord(cell, qp, 0);
      const MeshScalarT theta = sphere_coord(cell, qp, 1);
      coriolis(qp) = 2*Omega*( -cos(lambda)*cos(theta)*sin(alpha) + sin(theta)*cos(alpha));
    }
    
}
  

  
  
  
  
  
  
}
