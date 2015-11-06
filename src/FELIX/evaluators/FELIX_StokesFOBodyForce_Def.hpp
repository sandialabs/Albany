//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {

const double pi = 3.1415926535897932385;
double rho_g; //rho*g

//**********************************************************************

template<typename EvalT, typename Traits>
StokesFOBodyForce<EvalT, Traits>::
StokesFOBodyForce(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  force(p.get<std::string>("Body Force Variable Name"), dl->qp_vector),
  A(1.0),
  n(3.0),
  alpha(0.0)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  Teuchos::ParameterList* bf_list =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::ParameterList* p_list =
    p.get<Teuchos::ParameterList*>("Physical Parameter List");


  std::string type = bf_list->get("Type", "None");
  A = bf_list->get("Glen's Law A", 1.0);
  n = bf_list->get("Glen's Law n", 3.0);
  alpha = bf_list->get("FELIX alpha", 0.0);

  g = p_list->get("Gravity Acceleration", 9.8);
  rho = p_list->get("Ice Density", 910.0);
  rho_g = rho*g;

  stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  useStereographicMap = stereographicMapList->get("Use Stereographic Map", false);

#ifdef OUTPUT_TO_SCREEN
  *out << "rho, g: " << rho << ", " << g << std::endl;
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
    surfaceGrad = PHX::MDField<ScalarT,Cell,QuadPoint,Dim>(
             p.get<std::string>("Surface Height Gradient Name"), dl->qp_gradient);
    this->addDependentField(surfaceGrad);

    if(useStereographicMap) {
      surface = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string>("Surface Height Name"), dl->qp_scalar);
      this->addDependentField(surface);
      coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
                  p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
      this->addDependentField(coordVec);
    }
     bf_type = FO_INTERP_SURF_GRAD;
  }
#ifdef CISM_HAS_FELIX
  else if (type == "FO Surface Grad Provided") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Surface Grad Provided Source!" << std::endl;
#endif
    surfaceGrad = PHX::MDField<ScalarT,Cell,QuadPoint,Dim>(
             p.get<std::string>("Surface Height Gradient QP Variable Name"), dl->qp_gradient);
    this->addDependentField(surfaceGrad);
    bf_type = FO_SURF_GRAD_PROVIDED;
  }
#endif
  else if (type == "FOSinCos2D") {
    bf_type = FO_SINCOS2D;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  else if (type == "FOSinExp2D") {
    bf_type = FO_SINEXP2D;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  else if (type == "FOCosExp2D") {
    bf_type = FO_COSEXP2D;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  else if (type == "FOCosExp2DFlip") {
    bf_type = FO_COSEXP2DFLIP;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  else if (type == "FOCosExp2DAll") {
    bf_type = FO_COSEXP2DALL;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  else if (type == "FOSinCosZ") {
    bf_type = FO_SINCOSZ;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  else if (type == "Poisson") {
    bf_type = POISSON;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  //Source for xz MMS problem derived by Mauro.
  else if (type == "FO_XZ_MMS") {
    bf_type = FO_XZMMS;
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    this->addDependentField(muFELIX);
    this->addDependentField(coordVec);
  }
  //kept for backward compatibility. Use type = "FO INTERP GRAD SURF" instead.
  else if ((type == "FO ISMIP-HOM Test A") || (type == "FO ISMIP-HOM Test B") || (type == "FO ISMIP-HOM Test C") || (type == "FO ISMIP-HOM Test D")) {
  *out << "ISMIP-HOM Tests A/B/C/D \n WARNING: computing INTERP SURFACE GRAD Source! \nPlease set  Force Type = FO INTERP GRAD SURF." << std::endl;
    surfaceGrad = PHX::MDField<ScalarT,Cell,QuadPoint,Dim>(
        p.get<std::string>("Surface Height Gradient Name"), dl->qp_gradient);
    this->addDependentField(surfaceGrad);
    bf_type = FO_INTERP_SURF_GRAD;
  }
  else if (type == "FO Dome") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Dome Source!" << std::endl;
#endif
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_gradient);
    bf_type = FO_DOME;
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  vecDimFO = std::min(PHX::DataLayout::size_type(2), dims[2]); //vecDim (dims[2]) can be greater than 2 for coupled problems and = 1 for the problem in the xz plane


//*out << " in FELIX Stokes FO source! " << std::endl;
//*out << " vecDim = " << vecDim << std::endl;
//*out << " numDims = " << numDims << std::endl;
//*out << " numQPs = " << numQPs << std::endl;
//*out << " numNodes = " << numNodes << std::endl;

  this->setName("StokesFOBodyForce"+PHX::typeAsString<EvalT>());
}
//**********************************************************************
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_INTERP_SURF_GRAD_Tag& tag, const int& cell) const{

 if(useStereographicMap) {
       for (int qp=0; qp < numQPs; ++qp) {
         MeshScalarT x = coordVec(cell,qp,0)-x_0;
         MeshScalarT y = coordVec(cell,qp,1)-y_0;
         MeshScalarT z = coordVec(cell,qp,2);
         MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
         MeshScalarT h2 = h*h;
         MeshScalarT h_x = -x/2.0/R2*h2;
         MeshScalarT h_y = -y/2.0/R2*h2;
         force(cell,qp,0) = rho_g_kernel*(h*surfaceGrad(cell,qp,0) + (surface(cell,qp) - z) *(h_x-h_y)); //it already includes the integral weight h^2
         force(cell,qp,1) = rho_g_kernel*(h*surfaceGrad(cell,qp,1) + (surface(cell,qp) - z) *(h_y-h_x));
       }
   }
   else {
       for (int qp=0; qp < numQPs; ++qp) {
         force(cell,qp,0) = rho_g_kernel*surfaceGrad(cell,qp,0);
         force(cell,qp,1) = rho_g_kernel*surfaceGrad(cell,qp,1);
       }
   }


}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_SURF_GRAD_PROVIDED_Tag& tag, const int& cell) const{


}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const POISSON_Tag& tag, const int& cell) const{

   for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x = coordVec(cell,qp,0);
       force(cell,qp,0) = exp(x);
     }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_SINCOS2D_Tag& tag, const int& cell) const{

  double xphase=0.0, yphase=0.0;
  double r = 3*pi;
  for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muargt = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;
       MeshScalarT muqp = 0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       MeshScalarT dmuargtdx = -4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase);
       MeshScalarT dmuargtdy = -4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase);
       MeshScalarT exx = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;
       MeshScalarT eyy = -2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) - r;
       MeshScalarT exy = 0.0;
       force(cell,qp,0) = 2.0*muqp*(-4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy);
       force(cell,qp,1) = 2.0*muqp*(4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_COSEXP2D_Tag& tag, const int& cell) const{

  const double a = 1.0;
     for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       force(cell,qp,0) = 2.0*muqp*(2.0*a*a*exp(a*x)*cos(y2pi) + 6.0*pi*pi*cos(x2pi)*cos(y2pi) - 2.0*pi*pi*exp(a*x)*cos(y2pi));
       force(cell,qp,1) = 2.0*muqp*(-3.0*pi*a*exp(a*x)*sin(y2pi) - 10.0*pi*pi*sin(x2pi)*sin(y2pi));
     }


}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_COSEXP2DFLIP_Tag& tag, const int& cell) const{
    const double a = 1.0;
     for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       force(cell,qp,0) = 2.0*muqp*(-3.0*pi*a*exp(a*x)*sin(y2pi) - 10.0*pi*pi*sin(x2pi)*sin(y2pi));
       force(cell,qp,1) = 2.0*muqp*(1.0/2.0*a*a*exp(a*x)*cos(y2pi) + 6.0*pi*pi*cos(x2pi)*cos(y2pi) - 8.0*pi*pi*exp(a*x)*cos(y2pi));
     }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_COSEXP2DALL_Tag& tag, const int& cell) const{

  const double a = 1.0;
     for (int qp=0; qp < numQPs; ++qp) {
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
       force(cell,qp,0) = 2.0*muqp*(2.0*a*a*exp(a*x)*sin(y2pi) - 3.0*pi*a*exp(a*x)*sin(y2pi) - 2.0*pi*pi*exp(a*x)*sin(y2pi))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n-1.0)*pow(muargt, 1.0/n-2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy);
       force(cell,qp,1) = 2.0*muqp*(3.0*a*pi*exp(a*x)*cos(y2pi) + 1.0/2.0*a*a*exp(a*x)*cos(y2pi) - 8.0*pi*pi*exp(a*x)*cos(y2pi))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n-1.0)*pow(muargt, 1.0/n-2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
     }


}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_SINCOSZ_Tag& tag, const int& cell) const{

  for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT z = coordVec(cell,qp,2);
       MeshScalarT muqp = 1.0; //hard coded to constant for now

       ScalarT t1 = z*(1.0-z)*(1.0-2.0*z);
       ScalarT t2 = 2.0*z - 1.0;

       force(cell,qp,0) = 2.0*muqp*(-16.0*pi*pi*t1 + 3.0*t2)*sin(x2pi)*sin(y2pi);
       force(cell,qp,1) = 2.0*muqp*(16.0*pi*pi*t1 - 3.0*t2)*cos(x2pi)*cos(y2pi);
     }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_SINEXP2D_Tag& tag, const int& cell) const{

  for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       force(cell,qp,0) = -1.0*(-4.0*muqp*exp(coordVec(cell,qp,0)) - 12.0*muqp*pi*pi*cos(x2pi)*cos(y2pi) + 4.0*muqp*pi*pi*cos(y2pi));
       force(cell,qp,1) =  -1.0*(20.0*muqp*pi*pi*sin(x2pi)*sin(y2pi));
     }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_DOME_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y = coordVec(cell,qp,1);
       force(cell,qp,0) = -rho_g_kernel*x*0.7071/sqrt(450.0-x*x-y*y)/sqrt(450.0);
       force(cell,qp,1) = -rho_g_kernel*y*0.7071/sqrt(450.0-x*x-y*y)/sqrt(450.0);
     }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOBodyForce<EvalT, Traits>::
operator() (const FO_XZMMS_Tag& tag, const int& cell) const{

   double alpha0 = 4e-5; //renamed alpha alpha0 to prevent conflict with other alpha
   double s0 = 2.0;
   double H = 1.0;
   double beta = 1.0;
   //IK, 2/4/15, WARNING: I think the source term has been derived assuming n = 3, even
   //though in theory n is a free parameter...
   //TO DO: check sign!
     for (int qp=0; qp < numQPs; ++qp) {
       MeshScalarT x = coordVec(cell,qp,0); //x
       MeshScalarT z = coordVec(cell,qp,1); //z
       MeshScalarT s = s0 - alpha0*x*x;  //s = s0-alpha*x^2
       MeshScalarT phi1 = z - s; //phi1 = z-s
       //phi2 = 4*A*alpha^3*rho^3*g^3*x
       MeshScalarT phi2 = 4.0*A*pow(alpha0*rho_g_kernel, 3)*x;
       //phi3 = 4*x^3*phi1^5*phi2^2
       MeshScalarT phi3 = 4.0*x*x*x*pow(phi1,5)*phi2*phi2;
       //phi4 = 8*alpha*x^3*phi1^3*phi2 - (2*H*alpha*rho*g)/beta + 3*x*phi2*(phi1^4-H^4)
       MeshScalarT phi4 = 8.0*alpha0*pow(x*phi1,3)*phi2 - 2.0*H*alpha0*rho_g_kernel/beta
                        + 3.0*x*phi2*(pow(phi1,4) - pow(H,4));
       //phi5 = 56*alpha*x^2*phi1^3*phi2 + 48*alpha^2*x^4*phi1^2*phi2 + 6*phi2*(phi1^4-H^4
       MeshScalarT phi5 = 56.0*alpha0*x*x*pow(phi1,3)*phi2 + 48.0*alpha0*alpha0*pow(x,4)*phi1*phi1*phi2
                        + 6.0*phi2*(pow(phi1,4) - pow(H,4));
       //mu = 1/2*(A*phi4^2 + A*x*phi1*phi3)^(-1/3)
       MeshScalarT muargt = A*phi4*phi4 + A*x*phi1*phi3;
       MeshScalarT muqp = 0.5*pow(muargt, -1.0/3.0);
       //f = 16/3*A*mu^4*(-2*phi4^2*phi5 + 24*phi3*phi4*(phi1+2*alpha*x^2) - 6*x^3*phi1^2*phi2*phi3
       //                 -18*x^2*phi1^2*phi2*phi4^2 - 6*x*phi1*phi3*phi5);
       force(cell,qp,0) = 16.0/3.0*A*pow(muqp, 4)*(-2.0*phi4*phi4*phi5 + 24.0*phi3*phi4*(phi1 + 2.0*alpha0*x*x)
                                       -6.0*x*x*x*phi1*phi1*phi1*phi2*phi3 - 18.0*x*x*phi1*phi1*phi2*phi4*phi4
                                       -6.0*x*phi1*phi3*phi5);
     }

}
#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == FO_SINCOS2D || bf_type == FO_SINEXP2D || bf_type == FO_COSEXP2D || bf_type == FO_COSEXP2DFLIP ||
      bf_type == FO_COSEXP2DALL || bf_type == FO_SINCOSZ || bf_type == POISSON || bf_type == FO_XZMMS) {
    this->utils.setFieldData(muFELIX,fm);
    this->utils.setFieldData(coordVec,fm);
  }
  else if (bf_type == FO_DOME) {
    this->utils.setFieldData(coordVec,fm);
  }
  else if (bf_type == FO_INTERP_SURF_GRAD || bf_type == FO_SURF_GRAD_PROVIDED) {
    this->utils.setFieldData(surfaceGrad,fm);
    if(useStereographicMap) {
      this->utils.setFieldData(coordVec,fm);
      this->utils.setFieldData(surface,fm);
    }
  }

  this->utils.setFieldData(force,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell)
     for (std::size_t qp=0; qp < numQPs; ++qp)
       for (std::size_t i=0; i < vecDimFO; ++i)
     force(cell,qp,i) = 0.0;
 }
 //source using the gradient of the interpolated surface height
 else if (bf_type == FO_INTERP_SURF_GRAD || bf_type == FO_SURF_GRAD_PROVIDED) {
   if(useStereographicMap) {
     double R = stereographicMapList->get<double>("Earth Radius", 6371);
     double x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
     double y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
     double R2 = std::pow(R,2);
     for (std::size_t cell=0; cell < workset.numCells; ++cell) {
       for (std::size_t qp=0; qp < numQPs; ++qp) {
         MeshScalarT x = coordVec(cell,qp,0)-x_0;
         MeshScalarT y = coordVec(cell,qp,1)-y_0;
         MeshScalarT z = coordVec(cell,qp,2);
         MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
         MeshScalarT h2 = h*h;
         MeshScalarT h_x = -x/2.0/R2*h2;
         MeshScalarT h_y = -y/2.0/R2*h2;
         force(cell,qp,0) = rho_g*(h*surfaceGrad(cell,qp,0) + (surface(cell,qp) - z) *(h_x-h_y)); //it already includes the integral weight h^2
         force(cell,qp,1) = rho_g*(h*surfaceGrad(cell,qp,1) + (surface(cell,qp) - z) *(h_y-h_x));
       }
     }
   }
   else {
     for (std::size_t cell=0; cell < workset.numCells; ++cell) {
       for (std::size_t qp=0; qp < numQPs; ++qp) {
         force(cell,qp,0) = rho_g*surfaceGrad(cell,qp,0);
         force(cell,qp,1) = rho_g*surfaceGrad(cell,qp,1);
       }
     }
   }
 }
 else if (bf_type == FO_SINCOS2D) {
   double xphase=0.0, yphase=0.0;
   double r = 3*pi;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muargt = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;
       MeshScalarT muqp = 0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       MeshScalarT dmuargtdx = -4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase);
       MeshScalarT dmuargtdy = -4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase);
       MeshScalarT exx = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;
       MeshScalarT eyy = -2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) - r;
       MeshScalarT exy = 0.0;
       force(cell,qp,0) = 2.0*muqp*(-4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy);
       force(cell,qp,1) = 2.0*muqp*(4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
     }
   }
 }
 else if (bf_type == FO_SINEXP2D) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       force(cell,qp,0) = -1.0*(-4.0*muqp*exp(coordVec(cell,qp,0)) - 12.0*muqp*pi*pi*cos(x2pi)*cos(y2pi) + 4.0*muqp*pi*pi*cos(y2pi));
       force(cell,qp,1) =  -1.0*(20.0*muqp*pi*pi*sin(x2pi)*sin(y2pi));
     }
   }
 }
 else if (bf_type == POISSON) { //source term for debugging of Neumann BC
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x = coordVec(cell,qp,0);
       force(cell,qp,0) = exp(x);
     }
   }
 }
 else if  (bf_type == FO_XZMMS) { //source term for FO xz equations derived by Mauro
  //Hard-coding parameters here...
   double alpha0 = 4e-5; //renamed alpha alpha0 to prevent conflict with other alpha
   double s0 = 2.0;
   double H = 1.0;
   double beta = 1.0;
   //IK, 2/4/15, WARNING: I think the source term has been derived assuming n = 3, even
   //though in theory n is a free parameter...
   //TO DO: check sign!
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x = coordVec(cell,qp,0); //x
       MeshScalarT z = coordVec(cell,qp,1); //z
       MeshScalarT s = s0 - alpha0*x*x;  //s = s0-alpha*x^2
       MeshScalarT phi1 = z - s; //phi1 = z-s
       //phi2 = 4*A*alpha^3*rho^3*g^3*x
       MeshScalarT phi2 = 4.0*A*pow(alpha0*rho_g, 3)*x;
       //phi3 = 4*x^3*phi1^5*phi2^2
       MeshScalarT phi3 = 4.0*x*x*x*pow(phi1,5)*phi2*phi2;
       //phi4 = 8*alpha*x^3*phi1^3*phi2 - (2*H*alpha*rho*g)/beta + 3*x*phi2*(phi1^4-H^4)
       MeshScalarT phi4 = 8.0*alpha0*pow(x*phi1,3)*phi2 - 2.0*H*alpha0*rho_g/beta
                        + 3.0*x*phi2*(pow(phi1,4) - pow(H,4));
       //phi5 = 56*alpha*x^2*phi1^3*phi2 + 48*alpha^2*x^4*phi1^2*phi2 + 6*phi2*(phi1^4-H^4
       MeshScalarT phi5 = 56.0*alpha0*x*x*pow(phi1,3)*phi2 + 48.0*alpha0*alpha0*pow(x,4)*phi1*phi1*phi2
                        + 6.0*phi2*(pow(phi1,4) - pow(H,4));
       //mu = 1/2*(A*phi4^2 + A*x*phi1*phi3)^(-1/3)
       MeshScalarT muargt = A*phi4*phi4 + A*x*phi1*phi3;
       MeshScalarT muqp = 0.5*pow(muargt, -1.0/3.0);
       //f = 16/3*A*mu^4*(-2*phi4^2*phi5 + 24*phi3*phi4*(phi1+2*alpha*x^2) - 6*x^3*phi1^2*phi2*phi3
       //                 -18*x^2*phi1^2*phi2*phi4^2 - 6*x*phi1*phi3*phi5);
       force(cell,qp,0) = 16.0/3.0*A*pow(muqp, 4)*(-2.0*phi4*phi4*phi5 + 24.0*phi3*phi4*(phi1 + 2.0*alpha0*x*x)
                                       -6.0*x*x*x*phi1*phi1*phi1*phi2*phi3 - 18.0*x*x*phi1*phi1*phi2*phi4*phi4
                                       -6.0*x*phi1*phi3*phi5);
     }
   }
 }
 else if (bf_type == FO_COSEXP2D) {
   const double a = 1.0;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       force(cell,qp,0) = 2.0*muqp*(2.0*a*a*exp(a*x)*cos(y2pi) + 6.0*pi*pi*cos(x2pi)*cos(y2pi) - 2.0*pi*pi*exp(a*x)*cos(y2pi));
       force(cell,qp,1) = 2.0*muqp*(-3.0*pi*a*exp(a*x)*sin(y2pi) - 10.0*pi*pi*sin(x2pi)*sin(y2pi));
     }
   }
 }
 else if (bf_type == FO_COSEXP2DFLIP) {
   const double a = 1.0;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT muqp = 1.0 ; //0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       force(cell,qp,0) = 2.0*muqp*(-3.0*pi*a*exp(a*x)*sin(y2pi) - 10.0*pi*pi*sin(x2pi)*sin(y2pi));
       force(cell,qp,1) = 2.0*muqp*(1.0/2.0*a*a*exp(a*x)*cos(y2pi) + 6.0*pi*pi*cos(x2pi)*cos(y2pi) - 8.0*pi*pi*exp(a*x)*cos(y2pi));
     }
   }
 }
 else if (bf_type == FO_COSEXP2DALL) {
   const double a = 1.0;
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
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
       force(cell,qp,0) = 2.0*muqp*(2.0*a*a*exp(a*x)*sin(y2pi) - 3.0*pi*a*exp(a*x)*sin(y2pi) - 2.0*pi*pi*exp(a*x)*sin(y2pi))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n-1.0)*pow(muargt, 1.0/n-2.0)*(dmuargtdx*(2.0*exx + eyy) + dmuargtdy*exy);
       force(cell,qp,1) = 2.0*muqp*(3.0*a*pi*exp(a*x)*cos(y2pi) + 1.0/2.0*a*a*exp(a*x)*cos(y2pi) - 8.0*pi*pi*exp(a*x)*cos(y2pi))
            + 2.0*0.5*pow(A, -1.0/n)*(1.0/n-1.0)*pow(muargt, 1.0/n-2.0)*(dmuargtdx*exy + dmuargtdy*(exx + 2.0*eyy));
     }
   }
 }
 // Doubly-periodic MMS with polynomial in Z for FO Stokes
 else if (bf_type == FO_SINCOSZ) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       MeshScalarT z = coordVec(cell,qp,2);
       MeshScalarT muqp = 1.0; //hard coded to constant for now

       ScalarT t1 = z*(1.0-z)*(1.0-2.0*z);
       ScalarT t2 = 2.0*z - 1.0;

       force(cell,qp,0) = 2.0*muqp*(-16.0*pi*pi*t1 + 3.0*t2)*sin(x2pi)*sin(y2pi);
       force(cell,qp,1) = 2.0*muqp*(16.0*pi*pi*t1 - 3.0*t2)*cos(x2pi)*cos(y2pi);
     }
   }
 }
 //source for dome test case
 else if (bf_type == FO_DOME) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT y = coordVec(cell,qp,1);
       force(cell,qp,0) = -rho_g*x*0.7071/sqrt(450.0-x*x-y*y)/sqrt(450.0);
       force(cell,qp,1) = -rho_g*y*0.7071/sqrt(450.0-x*x-y*y)/sqrt(450.0);
     }
   }
 }
#else
  rho_g_kernel=rho_g;

  if (bf_type == NONE) {
    force.deep_copy(0.0);
  }
  else if (bf_type == FO_INTERP_SURF_GRAD || bf_type == FO_SURF_GRAD_PROVIDED) {
   R = stereographicMapList->get<double>("Earth Radius", 6371);
   x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
   y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
   R2 = std::pow(R,2);

   Kokkos::parallel_for(FO_INTERP_SURF_GRAD_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == POISSON) {
   Kokkos::parallel_for(POISSON_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_SINCOS2D) {
   Kokkos::parallel_for(FO_SINCOS2D_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_COSEXP2D) {
   Kokkos::parallel_for(FO_COSEXP2D_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_COSEXP2DFLIP) {
   Kokkos::parallel_for(FO_COSEXP2DFLIP_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_COSEXP2DALL) {
   Kokkos::parallel_for(FO_COSEXP2DALL_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_SINCOSZ) {
   Kokkos::parallel_for(FO_SINCOSZ_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_SINEXP2D) {
   Kokkos::parallel_for(FO_SINEXP2D_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_DOME) {
   Kokkos::parallel_for(FO_DOME_Policy(0,workset.numCells),*this);
  }
  else if (bf_type == FO_XZMMS ) {
   Kokkos::parallel_for(FO_XZMMS_Policy(0,workset.numCells),*this);
  }
#endif
}

} // Namespace FELIX
