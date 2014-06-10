//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>
#include <algorithm>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

// NINT(x) - nearest whole number
#define NINT(x) ( fabs(x)-fabs(int(x)) > 0.5 ) ? (x/fabs(x))*(int(fabs(x)+1)) : int(x)

// DIM(x,y) - gives x-y if this is positive and zero in the other case
# define DIM(x,y) (x-y > 0.0) ? x-y : 0.0 

namespace Aeras {
using Teuchos::rcp;
using PHX::MDALayout;


template<typename EvalT, typename Traits>
Atmosphere_Moisture<EvalT, Traits>::
Atmosphere_Moisture(Teuchos::ParameterList& p,
           const Teuchos::RCP<Aeras::Layouts>& dl) :
  coordVec        (p.get<std::string> ("QP Coordinate Vector Name"),     dl->qp_vector     ),
  Velx            (p.get<std::string> ("QP Velx"),                       dl->qp_scalar_level),
  Temp            (p.get<std::string> ("QP Temperature"),                dl->qp_scalar_level),
  Density         (p.get<std::string> ("QP Density"),                    dl->qp_scalar_level),
  Pressure        (p.get<std::string> ("QP Pressure"),                   dl->qp_scalar_level),
  Eta             (p.get<std::string> ("QP Eta"),                        dl->qp_scalar_level),
  TempSrc         (p.get<std::string> ("Temperature Source"),            dl->qp_scalar_level),
  tracerNames     (p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),
  tracerSrcNames(p.get< Teuchos::ArrayRCP<std::string> >("Tracer Source Names")),
  namesToSrc      (),
  numQPs          (dl->node_qp_scalar          ->dimension(2)),
  numDims         (dl->node_qp_gradient        ->dimension(3)),
  numLevels       (dl->node_scalar_level       ->dimension(2))
{  
  Teuchos::ArrayRCP<std::string> RequiredTracers(3);
  RequiredTracers[0] = "Vapor";
  RequiredTracers[1] = "Rain";
  RequiredTracers[2] = "Snow";
  for (int i=0; i<3; ++i) {
    bool found = false;
    for (int j=0; j<3 && !found; ++j)
      if (RequiredTracers[i] == tracerNames[j]) found = true;
    TEUCHOS_TEST_FOR_EXCEPTION(!found, std::logic_error,
      "Aeras::Atmosphere_Moisture requires Vapor, Rain and Snow tracers.");
  }

  this->addDependentField(coordVec);
  this->addDependentField(Velx);
  this->addDependentField(Density);
  this->addDependentField(Eta);
  this->addDependentField(Pressure);
  this->addDependentField(Temp);

  this->addEvaluatedField(TempSrc);

  for (int i = 0; i < tracerNames.size(); ++i) {
    namesToSrc[tracerNames[i]] = tracerSrcNames[i];
    PHX::MDField<ScalarT,Cell,QuadPoint> in   (tracerNames[i],   dl->qp_scalar_level);
    PHX::MDField<ScalarT,Cell,QuadPoint> src(tracerSrcNames[i],  dl->qp_scalar_level);
    TracerIn[tracerNames[i]]     = in;
    TracerSrc[tracerSrcNames[i]] = src;
    this->addDependentField(TracerIn   [tracerNames[i]]);
    this->addEvaluatedField(TracerSrc[tracerSrcNames[i]]);
  }
  this->setName("Aeras::Atmosphere_Moisture"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void Atmosphere_Moisture<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(Velx,    fm);
  this->utils.setFieldData(Temp,    fm);
  this->utils.setFieldData(Density, fm);
  this->utils.setFieldData(Pressure, fm);
  this->utils.setFieldData(Eta, fm);
  this->utils.setFieldData(TempSrc, fm);

  for (int i = 0; i < TracerIn.size();  ++i) this->utils.setFieldData(TracerIn[tracerNames[i]], fm);
  for (int i = 0; i < TracerSrc.size(); ++i) this->utils.setFieldData(TracerSrc[tracerSrcNames[i]],fm);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void Atmosphere_Moisture<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;
  //Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

   double dt_in = workset.current_time - workset.previous_time;
   double rainnc, rainncv;
   double zbot = 25.0;
   double ztop = 10000.0;

   std::vector<double> rho(numLevels, 0.0);
   std::vector<double> p(numLevels, 0.0);
   std::vector<double> t(numLevels, 0.0);
   std::vector<double> exner(numLevels, 0.0);
   std::vector<double> qv(numLevels, 0.0);
   std::vector<double> qc(numLevels, 0.0);
   std::vector<double> qr(numLevels, 0.0);
   std::vector<double> z(numLevels, 0.0);
   std::vector<double> dz8w(numLevels, 0.0);

  for (int i=0; i < TempSrc.size(); ++i) TempSrc(i)=0.0;

  for (int t=0; t < TracerSrc.size(); ++t)  
    for (int i=0; i < TracerSrc[tracerSrcNames[t]].size(); ++i) TracerSrc[tracerSrcNames[t]](i)=0.0;

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) { 

        rho[level]   = Albany::ADValue( Density(cell,qp,level) );
        p[level]     = Albany::ADValue( Pressure(cell,qp,level) );
        t[level]     = Albany::ADValue( Temp(cell,qp,level) );
        exner[level] = pow( (p[level]/1000.0),(0.286) );
        rho[level]   = Albany::ADValue( Density(cell,qp,level) );
        qv[level]    = Albany::ADValue( TracerIn["Vapor"](cell,qp,level) );
        qc[level]    = Albany::ADValue( TracerIn["Rain"](cell,qp,level) );
        qr[level]    = Albany::ADValue( TracerIn["Snow"](cell,qp,level) );
        z[level]     = (1.0-Albany::ADValue( Eta(cell,qp,level)) ) * ztop + zbot;
        dz8w[level]  = z[level];
      }

      kessler(numLevels, dt_in,
              rho, p, exner, dz8w,
              t, qv, qc, qr,
              rainnc,  rainncv,
              z);

      for (int level=0; level < numLevels; ++level) { 
        TracerSrc[namesToSrc["Vapor"]](cell,qp,level) += 0 * TracerIn["Vapor"] (cell,qp,level);
        TracerSrc[namesToSrc["Rain"]] (cell,qp,level) += 0 * TracerIn["Rain"]  (cell,qp,level);
        TracerSrc[namesToSrc["Snow"]] (cell,qp,level) += 0 * TracerIn["Snow"]  (cell,qp,level);
      }
    }
  }

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        TempSrc(cell,qp,level) += 0 * Temp(cell,qp,level);
      }
    }
  }

}

// **********************************************************************
template<typename EvalT, typename Traits>
void Atmosphere_Moisture<EvalT, Traits>::kessler(int Km, double dt_in,
             std::vector<double> & rho, 
             std::vector<double> & p, 
             std::vector<double> & exner, 
             std::vector<double> & dz8w,
             std::vector<double> & t,  
             std::vector<double> & qv, 
             std::vector<double> & qc, 
             std::vector<double> & qr,
             double &rainnc,  double &rainncv,
             std::vector<double> & z)
{

  int nfall, nfall_new;

  double xlv          = 2.501e+6; // Latent heat of vaporization at 0C [J/kg]
  double cp           = 1005.7;   // Specific heat capacity at constant pressure [J/kg/K]
  double Rd           = 287.04;   // Gas constant for dry air [J/kg/K]
  double Rv           = 461.5;    // Gas constant for water vapor [J/kg/K]
  double eps          = 0.622;    // epsilon, Ratio of Rd/Rv [unitless]
  double csvp3        = 29.65;    // Constant for saturation vapor pressure 
  double K_temp_C     = 273.15;   // Temperature in K at 0 C                     
  double mm_per_m     = 1000.;    // Convert, 1000 mm per m
  double mbar_per_bar = 1000.;    // Convert, 1000 mbar per bar 
  double mks_to_cgs   = 0.001;    // Convert mks to cgs
  double rhowater     = 1.0;      // Density of water [1g/cm3=1000kg/m3]

  double max_cr_sedimentation = 0.75;

  double qrprod;
  double qrvent;
  double qrevap;
  double gam;
  double qrr;     // Rainwater mixing ratio in the column [kg/kg] 
  double temp;    // temperature [K]
  double es;      // Saturation vapor pressure [mbar] from Bolton or Teton's formula 
  double qvs;     // Saturation mixing ratio (Rogers & Yau Eq. 2.18) [bar]
  double dz; 
  double dt;
  double f5; 
  double dtfall; 
  double rdz; 
  double prodct;
  double crmax; 
  double factorn; 
  double time_sediment; 
  double qcr; 
  double factorr; 
  double ppt;     // Precipitation at surface [m]

  std::vector<double> vt(Km, 0.0);
  std::vector<double> qrk(Km,0.0);
  std::vector<double> vtden(Km,0.0);
  std::vector<double> rdzk(Km,0.0);
  std::vector<double> rhok(Km,0.0);
  std::vector<double> rcgsk(Km,0.0);
  std::vector<double> factor(Km,0.0);
  std::vector<double> rdzw(Km,0.0);
  std::vector<double> qrcond(Km,1.0);

  dt = dt_in;
  f5 = 17.67*243.5*xlv/cp;    // changes?

   for (int k=0; k<Km; ++k) {             // construct column data
     qrk[k]   = qr[k];                // Save 3D rain to column
     rhok[k]  = rho[k];               // Save 3D dry air density to column
     rcgsk[k] = mks_to_cgs * rho[k];  // Save 3D dry air density to column
   }

  crmax = 0.0;

  // Set-up coefficients and compute stable timestep for
  // calculation of terminal velocity and vertical advection. 

  for (int k=0; k<Km; ++k) {    //do k = kts, kte

    qrr = std::max( 0.0,qrk[k]*rcgsk[k] );             // Total precip content 

    vtden[k] = sqrt(rcgsk[1]/rcgsk[k]);               // Kessler Eq. 4.3
    vt[k]    = 36.34 * pow(qrr,0.1364) * vtden[k];    // Kessler Eq. 8.11

    rdzw[k]  = 1.0/dz8w[k];

    crmax = std::max( vt[k]*dt*rdzw[k],crmax );  // Max precip speed in this column 
  } 

  for (int k=0; k<Km-1; ++k) { // do k = kts, kte-1               // Recompute ratio of vertical levels
    rdzk[k] = 1.0/(z[k+1] - z[k]);
  } 
  rdzk[Km-1] = 1.0/(z[Km-1] - z[Km-2]);

  // nint() - nearest whole number ???
  // nfall      = max(1,nint(0.5+crmax/max_cr_sedimentation))  
  nfall         = std::max( 1,int(0.5+crmax/max_cr_sedimentation) );  // courant number for big timestep.
  dtfall        = dt / double(nfall);                                 // splitting so courant number for sedimentation
  time_sediment = dt;                                                 // is stable

  // Calculate terminal velocity and vertical advection. 
  // Do a time split loop on this for stability.

  while ( nfall > 0 ) { //column_sedimentation: do while ( nfall > 0 )

    time_sediment = time_sediment - dtfall;
    for (int k=0; k<Km; ++k) {  //do k = kts, kte-1
      factor[k] = dtfall*rdzk[k]/rhok[k];
    } //enddo
    factor[Km-1] = dtfall*rdzk[Km-1];


    // Update cumulative large-scale prescipitation at surface
    ppt     = 0.;
    ppt     = rhok[0]*qrk[0]*vt[0]*dtfall/rhowater; 
    rainncv = ppt*mm_per_m;                          // convert units to mm
    rainnc  = rainnc + ppt*mm_per_m;                 // keep track of cumulative value


    // Time split loop, fallout done with flux upstream
    for (int k=0; k<Km; ++k) {    //do k = kts, kte-1
      qrk[k] = qrk[k] - factor[k] * ( rhok[k] * qrk[k] * vt[k] 
                                    - rhok[k+1] * qrk[k+1] * vt[k+1] );
    } 

    // Update rain at model top
    qrk[Km-1] = qrk[Km-1] - factor[Km-1]*qrk[Km-1]*vt[Km-1];

    // Compute new sedimentation velocity, and check/recompute new 
    // sedimentation timestep if this isn't the last split step.

    if( nfall > 1 ) { // this wasn't the last split sedimentation timestep

      nfall = nfall - 1;
      crmax = 0.0;
      
      for (int k=0; k<Km; ++k) { //do k = kts, kte 
        qrr   = std::max( 0.0,qrk[k]*rcgsk[k] );
        vt[k] = 36.34 * pow(qrr,0.1364) * vtden[k];
        crmax = std::max( vt[k]*time_sediment*rdzw[k],crmax );
      } // enddo

      //NINT - Macro for nearest whole number
      int nearwh = NINT( 0.5+crmax/max_cr_sedimentation );
      nfall_new = std::max( 1,nearwh );
      if (nfall_new != nfall ) {
        nfall  = nfall_new;
        dtfall = time_sediment/nfall;
      } 

    } else { // this was the last timestep

      for (int k=0; k<Km; ++k) { //do k=kts,kte
        qrcond[k] = qrk[k];
      }
      nfall = 0;  // exit condition for sedimentation loop

    }

  } //enddo column_sedimentation

  // Production of rain and deletion of qc
  // Production of qc from supersaturation
  // Evaporation of qr

  for (int k=0; k<Km; ++k) {  //do k = kts, kte
    factorn = 1.0 / (1.0+2.2*dt*std::max( 0.0,pow( qr[k],0.875 ) ));
    qrprod  = qc[k] * (1.0 - factorn)           
            + factorn*0.001*dt*std::max( qc[k]-0.001,0.0 );      
 
    qc[k] = std::max( qc[k]-qrprod,0.0 );
    qr[k] = (qr[k] + qrcond[k]-qr[k]);
    qr[k] = std::max( qr[k] + qrprod,0.0 );
 
    temp = exner[k]*t[k];          // Convert from potential temperature to temperature [K]
    temp = temp - K_temp_C;        // Convert from Kelvin to Celsius [C] 
 
    es  = 6.112 * exp( 17.67 * temp / (temp + 243.5)); // Saturation vapor pressure [mbar]
    es  = mbar_per_bar * es;                           // Saturation vapor pressure [bar]
    qvs = eps * es / (p[k] - es);                      // Saturation mixing ratio [bar]
 
 
    // Production of rain by condensation 
    qrcond[k] = (qv[k]-qvs) / (1.0 + p[k] / (p[k] - es)*qvs*f5/pow( (temp+243.5),2 ));
  
    // Ventilation factor
    qrvent = 1.6 + 124.9 * pow( (rcgsk[k] * qr[k]),0.2046 );
 
    // Evaporation of rain 
    double dim  = DIM( qvs,qv[k] );
    double arg1 = dt*((qrvent * pow( rcgsk[k]*qr[k],0.525 ))/(2.55E+08/(p[k]*qvs) + 5.4E+05))*(dim/(rcgsk[k]*qvs));
    double arg2 = std::max( -qrcond[k]-qc[k],0.0 );
    double arg3 = qr[k];
    qrevap      = std::min( arg1,arg2 );
    qrevap      = std::min( qrevap,arg3 );

    // Update all variables
    prodct    = std::max( qrcond[k],-qc[k] );
    gam       = xlv/(cp*exner[k]);
    t [k]     = t[k] + gam*(prodct - qrevap);
    qv[k]     = std::max( qv[k]-prodct+qrevap,0.0 );
    qc[k]     = qc[k] + prodct;
    qr[k]     = qr[k] - qrevap;
 
     //std::cout << "gam,prodct,qrevap: " << " " << gam << " " << prodct << " " << qrevap << std::endl;
     //std::cout << "rho,p,t,qv,qc,qr: " << rho[k] << " " << p[k] << " " << t[k] << " " << qv[k] << " " << qc[k] << " " << qr[k] << std::endl;
  } //enddo
  

  //for (int k=0; k<Km; ++k) {
  //  std::cout << "t: " << t[k] << std::endl;
  //}

}






}
