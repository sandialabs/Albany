//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {
const double pi = 3.1415926535897932385;
const double g = 9.8; //gravity for FELIX; hard-coded here for now
const double rho = 910; //density for FELIX; hard-coded here for now
const double rho_g = rho*g; //density for FELIX; hard-coded here for now

//**********************************************************************

template<typename EvalT, typename Traits>
StokesFOBodyForce<EvalT, Traits>::
StokesFOBodyForce(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  force(p.get<std::string>("Body Force Name"), dl->qp_vector), 
  A(1.0), 
  n(3.0), 
  alpha(0.0)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  A = bf_list->get("Glen's Law A", 1.0); 
  n = bf_list->get("Glen's Law n", 3.0); 
  alpha = bf_list->get("FELIX alpha", 0.0);
#ifdef OUTPUT_TO_SCREEN
  *out << "alpha: " << alpha << std::endl;
#endif 
  alpha *= pi/180.0; //convert alpha to radians 
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "FO INTERP SURF GRAD") {
#ifdef OUTPUT_TO_SCREEN
    *out << "INTERP SURFACE GRAD Source!" << std::endl;
#endif
    surfaceGrad = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
             p.get<std::string>("Surface Height Gradient Name"), dl->qp_gradient);
    this->addDependentField(surfaceGrad);
     bf_type = FO_INTERP_SURF_GRAD;
  }
  else if (type == "FOSinCos2D") {
    bf_type = FO_SINCOS2D;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "FOSinExp2D") {
    bf_type = FO_SINEXP2D;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "FOCosExp2D") {
    bf_type = FO_COSEXP2D;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "FOCosExp2DFlip") {
    bf_type = FO_COSEXP2DFLIP;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "FOCosExp2DAll") {
    bf_type = FO_COSEXP2DALL;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "FOSinCosZ") {
    bf_type = FO_SINCOSZ;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "Poisson") {
    bf_type = POISSON;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  //kept for backward compatibility. Use type = "FO INTERP GRAD SURF" instead.
  else if ((type == "FO ISMIP-HOM Test A") || (type == "FO ISMIP-HOM Test B") || (type == "FO ISMIP-HOM Test C") || (type == "FO ISMIP-HOM Test D")) {
	*out << "ISMIP-HOM Tests A/B/C/D \n WARNING: computing INTERP SURFACE GRAD Source! \nPlease set  Force Type = FO INTERP GRAD SURF." << std::endl;
    surfaceGrad = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
    		p.get<std::string>("Surface Height Gradient Name"), dl->qp_gradient);
    this->addDependentField(surfaceGrad);
    bf_type = FO_INTERP_SURF_GRAD;
  }
  else if (type == "FO Dome") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Dome Source!" << std::endl;
#endif 
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    bf_type = FO_DOME; 
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->qp_vector->dimensions(dims);
  vecDim  = dims[2];

//*out << " in FELIX Stokes FO source! " << std::endl;
//*out << " vecDim = " << vecDim << std::endl;
//*out << " numDims = " << numDims << std::endl;
//*out << " numQPs = " << numQPs << std::endl; 

  this->setName("StokesFOBodyForce"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == FO_SINCOS2D || bf_type == FO_SINEXP2D || bf_type == FO_COSEXP2D || bf_type == FO_COSEXP2DFLIP || 
      bf_type == FO_COSEXP2DALL || bf_type == FO_SINCOSZ || bf_type == POISSON) {
    this->utils.setFieldData(muFELIX,fm);
    this->utils.setFieldData(coordVec,fm);
  }
  else if (bf_type == FO_DOME) {
    this->utils.setFieldData(coordVec,fm);
  }
  else if (bf_type == FO_INTERP_SURF_GRAD)
	  this->utils.setFieldData(surfaceGrad,fm);

  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < vecDim; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
 //source using the gradient of the interpolated surface height
 else if (bf_type == FO_INTERP_SURF_GRAD) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       force(cell,qp,0) = rho_g*surfaceGrad(cell,qp,0);
       force(cell,qp,1) = rho_g*surfaceGrad(cell,qp,1);
     }
   }
 }
 else if (bf_type == FO_SINCOS2D) {
   double xphase=0.0, yphase=0.0;
   double r = 3*pi;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muargt = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;  
       MeshScalarT muqp = 0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       MeshScalarT dmuargtdx = -4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase); 
       MeshScalarT dmuargtdy = -4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase); 
       MeshScalarT exx = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r; 
       MeshScalarT eyy = -2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) - r; 
       MeshScalarT exy = 0.0;  
       f[0] = 2.0*muqp*(-4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase))  
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy); 
       f[1] = 2.0*muqp*(4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase)) 
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
     }
   }
 }
 else if (bf_type == FO_SINEXP2D) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       f[0] = -1.0*(-4.0*muqp*exp(coordVec(cell,qp,0)) - 12.0*muqp*pi*pi*cos(x2pi)*cos(y2pi) + 4.0*muqp*pi*pi*cos(y2pi));   
       f[1] =  -1.0*(20.0*muqp*pi*pi*sin(x2pi)*sin(y2pi)); 
     }
   }
 }
 else if (bf_type == POISSON) { //source term for debugging of Neumann BC
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       f[0] = exp(x);  
     }
   }
 }
 else if (bf_type == FO_COSEXP2D) {
   const double a = 1.0; 
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       f[0] = 2.0*muqp*(2.0*a*a*exp(a*x)*cos(y2pi) + 6.0*pi*pi*cos(x2pi)*cos(y2pi) - 2.0*pi*pi*exp(a*x)*cos(y2pi)); 
       f[1] = 2.0*muqp*(-3.0*pi*a*exp(a*x)*sin(y2pi) - 10.0*pi*pi*sin(x2pi)*sin(y2pi)); 
     }
   }
 }
 else if (bf_type == FO_COSEXP2DFLIP) {
   const double a = 1.0; 
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       f[0] = 2.0*muqp*(-3.0*pi*a*exp(a*x)*sin(y2pi) - 10.0*pi*pi*sin(x2pi)*sin(y2pi)); 
       f[1] = 2.0*muqp*(1.0/2.0*a*a*exp(a*x)*cos(y2pi) + 6.0*pi*pi*cos(x2pi)*cos(y2pi) - 8.0*pi*pi*exp(a*x)*cos(y2pi)); 
     }
   }
 }
 else if (bf_type == FO_COSEXP2DALL) {
   const double a = 1.0; 
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muargt = (a*a + 4.0*pi*pi - 2.0*pi*a)*sin(y2pi)*sin(y2pi) + 1.0/4.0*(2.0*pi+a)*(2.0*pi+a)*cos(y2pi)*cos(y2pi);
       muargt = sqrt(muargt)*exp(a*x);
       MeshScalarT muqp = 1.0/2.0*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       MeshScalarT dmuargtdx = a*muargt;
       MeshScalarT dmuargtdy = 3.0/2.0*pi*(a*a+4.0*pi*pi-4.0*pi*a)*cos(y2pi)*sin(y2pi)*exp(a*x)/sqrt((a*a + 4.0*pi*pi - 2.0*pi*a)*sin(y2pi)*sin(y2pi) + 1.0/4.0*(2.0*pi+a)*(2.0*pi+a)*cos(y2pi)*cos(y2pi));
       MeshScalarT exx = a*exp(a*x)*sin(y2pi);
       MeshScalarT eyy = -2.0*pi*exp(a*x)*sin(y2pi);
       MeshScalarT exy = 1.0/2.0*(2.0*pi+a)*exp(a*x)*cos(y2pi);
       f[0] = 2.0*muqp*(2.0*a*a*exp(a*x)*sin(y2pi) - 3.0*pi*a*exp(a*x)*sin(y2pi) - 2.0*pi*pi*exp(a*x)*sin(y2pi))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n-1.0)*pow(muargt, 1.0/n-2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy);
       f[1] = 2.0*muqp*(3.0*a*pi*exp(a*x)*cos(y2pi) + 1.0/2.0*a*a*exp(a*x)*cos(y2pi) - 8.0*pi*pi*exp(a*x)*cos(y2pi))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n-1.0)*pow(muargt, 1.0/n-2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
     }
   }
 }
 // Doubly-periodic MMS with polynomial in Z for FO Stokes
 else if (bf_type == FO_SINCOSZ) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT& z = coordVec(cell,qp,2);
       MeshScalarT muqp = 1.0; //hard coded to constant for now 
       
       ScalarT t1 = z*(1.0-z)*(1.0-2.0*z); 
       ScalarT t2 = 2.0*z - 1.0; 

       f[0] = 2.0*muqp*(-16.0*pi*pi*t1 + 3.0*t2)*sin(x2pi)*sin(y2pi); 
       f[1] = 2.0*muqp*(16.0*pi*pi*t1 - 3.0*t2)*cos(x2pi)*cos(y2pi);  
     }
   }
 }
 //source for dome test case 
 else if (bf_type == FO_DOME) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       ScalarT* f = &force(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y = coordVec(cell,qp,1);
       f[0] = -rho_g*x*0.7071/sqrt(450.0-x*x-y*y)/sqrt(450.0);  
       f[1] = -rho_g*y*0.7071/sqrt(450.0-x*x-y*y)/sqrt(450.0);  
     }
   }
 }
}

}

