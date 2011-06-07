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
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_AbstractDiscretization.hpp"


namespace QCAD {

template<typename EvalT,typename Traits>
PoissonDirichlet<EvalT, Traits>::
PoissonDirichlet(Teuchos::ParameterList& p) :
  PHAL::Dirichlet<EvalT,Traits>(p)
{
  user_value = p.get<RealType>("Dirichlet Value");
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Poisson Source Parameter List");

  carrierStatistics = psList->get("Carrier Statistics", "Boltzmann Statistics");
  incompIonization = psList->get("Incomplete Ionization", "False");
  dopingDonor = psList->get("Donor Doping", 1e14);
  dopingAcceptor = psList->get("Acceptor Doping", 1e14);
  donorActE = psList->get("Donor Activation Energy", 0.040);
  acceptorActE = psList->get("Acceptor Activation Energy", 0.045);
 
  temperature = p.get<double>("Temperature"); //To be replaced by SharedParameter evaluator access


  double affinitySetByUser, affinityOfDOF;
  std::string materialSetByUser;
  std::string materialOfDOF;
  std::string nodeSetName = PHAL::DirichletBase<EvalT,Traits>::nodeSetID;

  //! Get offset due to electon affinity difference only once in constructor
  materialSetByUser = materialDB->getNodeSetParam<string>(nodeSetName,"materialSetByUser","");
  materialOfDOF = materialDB->getNodeSetParam<string>(nodeSetName,"materialOfDOF","");

  if( materialSetByUser.length() == 0) {
    this->ebSetByUser = materialDB->getNodeSetParam<string>(nodeSetName,"elementBlockSetByUser","");
    if( ebSetByUser.length() > 0 )
      affinitySetByUser = materialDB->getElementBlockParam<double>(ebSetByUser,"Electron Affinity");
    else affinitySetByUser = 0; //default case when there are no parameters
  }
  else 
    affinitySetByUser = materialDB->getMaterialParam<double>(materialSetByUser,
							     "Electron Affinity");

  if( materialOfDOF.length() == 0 ) {
    this->ebOfDOF = materialDB->getNodeSetParam<string>(nodeSetName,"elementBlockOfDOF","");
    if( ebOfDOF.length() > 0) 
      affinityOfDOF = materialDB->getElementBlockParam<double>(ebOfDOF,"Electron Affinity");
    else affinityOfDOF = 0; //default case when there are no parameters
  }
  else 
    affinityOfDOF = materialDB->getMaterialParam<double>(materialOfDOF,
							 "Electron Affinity");

  this->offsetDueToAffinity = affinitySetByUser - affinityOfDOF;
}

template<typename EvalT,typename Traits>
void PoissonDirichlet<EvalT, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  double kbBoltz = 8.617343e-05; //[eV/K]
  ScalarT V0 = kbBoltz*temperature/1.0; // kb*T/q in [V], scaling for potential
  ScalarT phiForIncompleteIon;

  //! Get offset due to doping in semiconductors
  ScalarT offsetDueToDoping = 0.0;
  if( ebSetByUser.length() > 0) {
    phiForIncompleteIon = user_value/V0;
    offsetDueToDoping -= ComputeOffsetDueToDoping(ebSetByUser, phiForIncompleteIon); 
  }

  if( ebOfDOF.length() > 0 ) {
    phiForIncompleteIon =  (user_value+offsetDueToAffinity)/V0;
    offsetDueToDoping += ComputeOffsetDueToDoping(ebOfDOF,phiForIncompleteIon);
  }

  // Suzey: phi values correct?

  ScalarT newValue = (user_value + offsetDueToAffinity) * 1.0/V0 + offsetDueToDoping;
  PHAL::DirichletBase<EvalT,Traits>::value = newValue;

  //std::cout << "DEBUG: User value " << user_value << " --> " << newValue
  //	    << " (V0=" << V0 << ", offsetA="<<offsetDueToAffinity
  //	    << ", offsetD="<<offsetDueToDoping<<")" << std::endl;

  //! Call base class evaluateFields, which sets relevant nodes using value member
  PHAL::Dirichlet<EvalT,Traits>::evaluateFields(dirichletWorkset);
}




template<typename EvalT,typename Traits>
typename PoissonDirichlet<EvalT,Traits>::ScalarT
PoissonDirichlet<EvalT, Traits>::
ComputeOffsetDueToDoping(const std::string ebName, ScalarT phiForIncompleteIon)
{
  const double kbBoltz = 8.617343e-05; // [eV/K]
  const double m0 = 9.10938215e-31;    // [kg]
  const double pi = 3.141592654;
  const double hbar = 1.054571628e-34; // [J.s]

  ScalarT offsetDueToDoping = 0.0;

  //! Similar logic as QCAD::PoissonSource::evaluateFields_elementblocks -- consolidate?

  string matrlCategory = materialDB->getElementBlockParam<string>(ebName,"Category");
  if(matrlCategory == "Semiconductor") {
   
    //! temperature-independent material parameters
    double mdn = materialDB->getElementBlockParam<double>(ebName,"Electron DOS Effective Mass");
    double mdp = materialDB->getElementBlockParam<double>(ebName,"Hole DOS Effective Mass");
    double Tref = materialDB->getElementBlockParam<double>(ebName,"Reference Temperature");
    
    double Eg0 = materialDB->getElementBlockParam<double>(ebName,"Zero Temperature Band Gap");
    double alpha = materialDB->getElementBlockParam<double>(ebName,"Band Gap Alpha Coefficient");
    double beta = materialDB->getElementBlockParam<double>(ebName,"Band Gap Beta Coefficient");
    
    // constant prefactor in calculating Nc and Nv in [cm-3]
    double NcvFactor = 2.0*pow((kbBoltz*1.602e-19*m0*Tref)/(2*pi*pow(hbar,2)),3./2.)*1e-6;
            // 1.602e-19 converts kbBoltz in [eV/K] to [J/K], 1e-6 converts [m-3] to [cm-3]
    
    //! strong temperature-dependent material parameters
    ScalarT Nc;  // conduction band effective DOS in [cm-3]
    ScalarT Nv;  // valence band effective DOS in [cm-3]
    ScalarT Eg;  // band gap at T [K] in [eV]
    ScalarT Eic; // intrinsic Fermi level - conduction band edge in [eV]
    ScalarT Evi; // valence band edge - intrinsic Fermi level in [eV]
    
    Nc = NcvFactor*pow(mdn,1.5)*pow(temperature/Tref,1.5);  // in [cm-3]
    Nv = NcvFactor*pow(mdp,1.5)*pow(temperature/Tref,1.5); 
    Eg = Eg0-alpha*pow(temperature,2.0)/(beta+temperature); // in [eV]
    
    ScalarT kbT = kbBoltz*temperature;      // in [eV]
    Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [eV]
    Evi = -Eg/2. - 3./4.*kbT*log(mdp/mdn);  // (Ev-Ei) in [eV]
    
    //! function pointer to carrier statistics member function
    ScalarT (QCAD::PoissonDirichlet<EvalT,Traits>::*invCarrStat) (const ScalarT);
    
    if (carrierStatistics == "Boltzmann Statistics")
      invCarrStat = &QCAD::PoissonDirichlet<EvalT,Traits>::invComputeMBStat;  
    else if (carrierStatistics == "Fermi-Dirac Statistics")
      invCarrStat = &QCAD::PoissonDirichlet<EvalT,Traits>::invComputeFDIntOneHalf;
    else if (carrierStatistics == "0-K Fermi-Dirac Statistics")
      invCarrStat = &QCAD::PoissonDirichlet<EvalT,Traits>::invComputeZeroKFDInt;
    else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Unknown carrier statistics ! " << std::endl);

    
    //! function pointer to ionized dopants member function
    ScalarT (QCAD::PoissonDirichlet<EvalT,Traits>::*ionDopant) (const std::string, const ScalarT&); 
    if (incompIonization == "False")
      ionDopant = &QCAD::PoissonDirichlet<EvalT,Traits>::fullDopants; 
    else if (incompIonization == "True")
      ionDopant = &QCAD::PoissonDirichlet<EvalT,Traits>::ionizedDopants;
    else TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Invalid incomplete ionization option ! " << std::endl);


    //! Dopant type and profile
    string dopantType = materialDB->getElementBlockParam<string>(ebName,"dopantType","None");
    string dopingProfile = materialDB->getElementBlockParam<string>(ebName,"dopingProfile","Constant");

    double dopantActE, dopingConc;
    if(dopantType == "Donor") {
      dopingConc = dopingDonor;
      dopantActE = donorActE;
    }
    else if(dopantType == "Acceptor") {
      dopingConc = dopingAcceptor;
      dopantActE = acceptorActE;
    }
    else if(dopantType == "None") {
      dopingConc = dopantActE = 0.0;
    }
    else {
      TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
			  std::endl << "Error!  Unknown dopant type " << dopantType << "!"<< std::endl);
    }

    //! determine the ionized donor concentration
    ScalarT ionN, inArg;
    inArg = phiForIncompleteIon + Eic/kbT + dopantActE/kbT; //Suzey: is this correct for n- and p-type?
    ionN = (this->*ionDopant)(dopantType,inArg)*dopingConc;

    //! compute the potential offset due to semiconductor doping
    if(ionN < 0) offsetDueToDoping =  1.0 * (this->*invCarrStat)( -ionN/Nv ) + Evi/kbT;
    else         offsetDueToDoping = -1.0 * (this->*invCarrStat)(  ionN/Nc ) - Eic/kbT;

    //std::cout << "DEBUG: OffsetDueToDoping for " << ebName << ": ionN=" 
    //      << ionN << " offset=" << offsetDueToDoping << std::endl;
  }
  return offsetDueToDoping;
}


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::invComputeMBStat(const ScalarT x)
{
   return -log(x);
}


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::invComputeFDIntOneHalf(const ScalarT x)
{
  //Suzey TODO - add inverse of commented function below, for now I just copied boltzmann case (egn)
  return -log(x);

  /*
   // Use the approximate 1/2 FD integral D. Bednarczyk and J. Bednarczyk, 
   // "The approximation of the Fermi-Dirac integral F_{1/2}(x),"
   // Physics Letters A, vol.64, no.4, pp.409-410, 1978. The approximation 
   // has error < 0.4% in the entire x range.  
   
   ScalarT fdInt; 
   if (x >= -50.0)
   {
     fdInt = pow(x,4.) + 50. + 33.6*x*(1.-0.68*exp(-0.17*pow((x+1.),2.0)));
     fdInt = pow((exp(-x) + (3./4.*sqrt(pi)) * pow(fdInt, -3./8.)),-1.0);
   }      
   else
     fdInt = exp(x); // for x<-50, the 1/2 FD integral is well approximated by exp(x)
     
   return fdInt;
  */
}


// **********************************************************************
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::invComputeZeroKFDInt(const ScalarT x)
{
  //Suzey TODO - add inverse of commented function below, for now I just copied boltzmann case (egn)
  return -log(x);

  /*
   ScalarT zeroKFDInt;
   if (x > 0.0) 
     zeroKFDInt = 4./3./sqrt(pi)*pow(x, 3./2.);
   else
     zeroKFDInt = 0.0;
      
   return zeroKFDInt;
  */
}


// **********************************************************************
//Copied from QCAD::PoissonSource
template<typename EvalT,typename Traits>
inline typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::fullDopants(const std::string dopType, const ScalarT &x)
{
  ScalarT ionDopants;

  // fully ionized (create function to use function pointer)
  if (dopType == "Donor")
    ionDopants = 1.0;
  else if (dopType == "Acceptor")
    ionDopants = -1.0;
  else if (dopType == "None")
    ionDopants = 0.0;
  else
  {
    ionDopants = 0.0;
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }


  return ionDopants;  
}


// **********************************************************************
//Copied from QCAD::PoissonSource
template<typename EvalT,typename Traits>
typename QCAD::PoissonDirichlet<EvalT,Traits>::ScalarT
QCAD::PoissonDirichlet<EvalT,Traits>::ionizedDopants(const std::string dopType, const ScalarT &x)
{
  ScalarT ionDopants;
   
  if (dopType == "Donor")
    ionDopants = 1.0 / (1. + 2.*exp(x));  
  else if (dopType == "Acceptor")
    ionDopants = -1.0 / (1. + 4.*exp(x));
  else if (dopType == "None")
    ionDopants = 0.0;
  else
  {
    ionDopants = 0.0;
    TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error!  Unknown dopant type " << dopType << "!"<< std::endl);
  }
   
  return ionDopants; 
}



}  //end of QCAD namespace
