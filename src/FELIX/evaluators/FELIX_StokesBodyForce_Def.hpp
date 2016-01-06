//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace FELIX {
const double pi = 3.1415926535897932385;
const double g = 9.8; //gravity for FELIX; hard-coded here for now
const double rho = 910; //density for FELIX; hard-coded here for now
const double L = 5; 
const double alpha = 0.5*pi/180; 
const double Z = 1; 
const double U = L*pow(2*rho*g*Z,3); 
const double cx = 1.0; //1e-9*U; 
const double cy = 1.0; //1e-9*U; 

//**********************************************************************
//function returning u solution to Poly source test case
template<typename ScalarT>
ScalarT u2d(const ScalarT& x, const ScalarT& y){
  ScalarT u = 20.0*(x-1.0)*(x-1.0)*x*x*y*(2.0*y*y-3.0*y+1.0);
  return u; 
}

//function returning v solution to Poly source test case
template<typename ScalarT>
ScalarT v2d(const ScalarT& x, const ScalarT& y){
  ScalarT v = -20.0*x*(2.0*x*x-3.0*x+1.0)*y*y*(1.0-y)*(1.0-y); 
  return v; 
}

//function returning p solution to Poly source test case
template<typename ScalarT>
ScalarT p2d(const ScalarT& x, const ScalarT& y){
  ScalarT p = (5.0/2.0*y*y-10.0*x) + 4.1613;
  return p; 
}

//function returning tau11 stress for Poly source term test case
template<typename ScalarT>
ScalarT tau112d(const ScalarT& x, const ScalarT& y){
  int num_deriv = 1;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(y); 
  Sacado::Fad::DFad<ScalarT> ufad = u2d(xfad, yfad); 
  ScalarT dudx_ad = ufad.dx(0); 
  ScalarT p = p2d(x,y); 
  ScalarT tau = 2.0*dudx_ad - p;   
  return tau; 
}

//function returning tau12 stress for Poly source term test case
template<typename ScalarT>
ScalarT tau122d(const ScalarT& x, const ScalarT& y){
  int num_deriv = 2;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, y); 
  Sacado::Fad::DFad<ScalarT> ufad = u2d(xfad, yfad); 
  Sacado::Fad::DFad<ScalarT> vfad = v2d(xfad, yfad); 
  ScalarT dudy_ad = ufad.dx(1); 
  ScalarT dvdx_ad = vfad.dx(0); 
  ScalarT tau = (dudy_ad + dvdx_ad);   
  return tau; 
}

//function returning tau22 stress for Poly source term test case
template<typename ScalarT>
ScalarT tau222d(const ScalarT& x, const ScalarT& y){
  int num_deriv = 1;  
  Sacado::Fad::DFad<ScalarT> xfad(x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 0, y); 
  Sacado::Fad::DFad<ScalarT> vfad = v2d(xfad, yfad); 
  ScalarT dvdy_ad = vfad.dx(0); 
  ScalarT p = p2d(x,y);  
  ScalarT tau = 2.0*dvdy_ad - p;   
  return tau; 
}

//returns function giving top surface boundary for MMF Test A ISMIP-HOM test case
template<typename ScalarT>
ScalarT top_surf(const ScalarT& x, const ScalarT& y){
  ScalarT s = -x*tan(alpha); 
  return s; 
}

//returns function basal surface boundary for MMF TestA ISMIP-HOM test case
template<typename ScalarT>
ScalarT basal_surf(const ScalarT& x, const ScalarT& y){
  ScalarT s = top_surf(x, y);
  ScalarT b = s + Z/2.0*sin(2.0*pi*x/L)*sin(2.0*pi*y/L) - Z; 
  return b; 
}

//returns expression for u-velocity for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT u_vel(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  ScalarT b = basal_surf(x, y); 
  ScalarT s = top_surf(x, y); 
  ScalarT a = (s-z)/(s-b); 
  ScalarT u = cx*(1.0-a*a*a*a); 
  return u; 
}


//returns expression for v-velocity for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT v_vel(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  ScalarT b = basal_surf(x, y); 
  ScalarT s = top_surf(x, y); 
  ScalarT a = (s-z)/(s-b); 
  ScalarT a4 = a*a*a*a; 
  ScalarT v = cy/(s-b)*(1.0-a4) - 0.5*cx/(s-b)*(1.0-a4)*Z*cos(2.0*pi*x/L)*cos(2*pi*y/L); //exp(ct*t) term??
  return v; 
}

//returns expression for w-velocity for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT w_vel(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 2;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, y); 
  Sacado::Fad::DFad<ScalarT> bfad = basal_surf(xfad, yfad); 
  Sacado::Fad::DFad<ScalarT> sfad = top_surf(xfad, yfad); 
  ScalarT dbdx_ad = bfad.dx(0); 
  ScalarT dbdy_ad = bfad.dx(1); 
  ScalarT dsdx_ad = sfad.dx(0); 
  ScalarT dsdy_ad = sfad.dx(1); 
  ScalarT u = u_vel(x,y,z); 
  ScalarT v = v_vel(x,y,z); 
  ScalarT s = top_surf(x,y); 
  ScalarT b = basal_surf(x,y);
  ScalarT w = u*(dbdx_ad*(s-z)/(s-b) + dsdx_ad*(z-b)/(s-b)) + 
              v*(dbdy_ad*(s-z)/(s-b) + dsdy_ad*(z-b)/(s-b)); 
  return w; 
}

//returns expression for mu for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT mu_eval(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  /*int num_deriv = 3;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, y); 
  Sacado::Fad::DFad<ScalarT> zfad(num_deriv, 2, y);
  Sacado::Fad::DFad<ScalarT> ufad = u_vel(xfad, yfad, zfad); 
  Sacado::Fad::DFad<ScalarT> vfad = v_vel(xfad, yfad, zfad); 
  Sacado::Fad::DFad<ScalarT> wfad = w_vel(xfad, yfad, zfad); 
  ScalarT dudx_ad = ufad.dx(0); 
  ScalarT dudy_ad = ufad.dx(1); 
  ScalarT dudz_ad = ufad.dx(2); 
  ScalarT dvdx_ad = vfad.dx(0); 
  ScalarT dvdy_ad = vfad.dx(1); 
  ScalarT dvdz_ad = vfad.dx(2); 
  ScalarT dwdx_ad = wfad.dx(0); 
  ScalarT dwdy_ad = wfad.dx(1); 
  ScalarT dwdz_ad = wfad.dx(2); 
  ScalarT mu = 0.25*(dudy_ad + dvdx_ad)*(dudy_ad + dvdx_ad) + 
               0.25*(dudz_ad + dwdx_ad)*(dudz_ad + dwdx_ad) + 
               0.25*(dvdz_ad + dwdy_ad)*(dvdz_ad + dwdy_ad) - 
               dudx_ad*dvdy_ad - dudx_ad*dwdz_ad - dvdy_ad*dwdz_ad; 
  mu = pow(mu, (1.0-n)/(2.0*n)); 
  mu *= 0.25*pow(A, -1.0/n);
  */ 
  ScalarT mu = 1.0;  
  return mu; 
}

//returns expression for pressure for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT press(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 2;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, y); 
  Sacado::Fad::DFad<ScalarT> zfad(z); 
  Sacado::Fad::DFad<ScalarT> ufad = u_vel(xfad, yfad, zfad); 
  Sacado::Fad::DFad<ScalarT> vfad = v_vel(xfad, yfad, zfad); 
  ScalarT dudx_ad = ufad.dx(0); 
  ScalarT dvdy_ad = vfad.dx(1);
  ScalarT mu = mu_eval(x, y, z); 
  ScalarT s = top_surf(x,y); 
  ScalarT p = 2.0*mu*dudx_ad + 2.0*mu*dvdy_ad - rho*g*(s-z);  
  return p; 
}

//returns expression for tau11 for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT tau11(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 1;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(y); 
  Sacado::Fad::DFad<ScalarT> zfad(z); 
  Sacado::Fad::DFad<ScalarT> ufad = u_vel(xfad, yfad, zfad); 
  ScalarT dudx_ad = ufad.dx(0); 
  ScalarT mu = mu_eval(x, y, z); 
  ScalarT p = press(x,y,z); 
  ScalarT tau = 2.0*mu*dudx_ad - p;   
  return tau; 
}

//returns expression for tau12 for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT tau12(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 2;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, y); 
  Sacado::Fad::DFad<ScalarT> zfad(z); 
  Sacado::Fad::DFad<ScalarT> ufad = u_vel(xfad, yfad, zfad); 
  Sacado::Fad::DFad<ScalarT> vfad = v_vel(xfad, yfad, zfad); 
  ScalarT dudy_ad = ufad.dx(1); 
  ScalarT dvdx_ad = vfad.dx(0); 
  ScalarT mu = mu_eval(x, y, z); 
  ScalarT tau = mu*(dudy_ad + dvdx_ad);   
  return tau; 
}

//returns expression for tau13 for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT tau13(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 2;  
  Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, x);
  Sacado::Fad::DFad<ScalarT> zfad(num_deriv, 1, z); 
  Sacado::Fad::DFad<ScalarT> yfad(y); 
  Sacado::Fad::DFad<ScalarT> ufad = u_vel(xfad, yfad, zfad); 
  Sacado::Fad::DFad<ScalarT> wfad = w_vel(xfad, yfad, zfad); 
  ScalarT dudz_ad = ufad.dx(1); 
  ScalarT dwdx_ad = wfad.dx(0); 
  ScalarT mu = mu_eval(x, y, z); 
  ScalarT tau = mu*(dudz_ad + dwdx_ad);   
  return tau; 
}

//returns expression for tau22 for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT tau22(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 1;  
  Sacado::Fad::DFad<ScalarT> xfad(x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 0, y); 
  Sacado::Fad::DFad<ScalarT> zfad(z); 
  Sacado::Fad::DFad<ScalarT> vfad = v_vel(xfad, yfad, zfad); 
  ScalarT dvdy_ad = vfad.dx(0); 
  ScalarT mu = mu_eval(x, y, z);
  ScalarT p = press(x,y,z);  
  ScalarT tau = mu*(2.0*mu*dvdy_ad - p);   
  return tau; 
}

//returns expression for tau23 for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT tau23(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 2;  
  Sacado::Fad::DFad<ScalarT> xfad(x);
  Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 0, y); 
  Sacado::Fad::DFad<ScalarT> zfad(num_deriv, 1, z); 
  Sacado::Fad::DFad<ScalarT> vfad = v_vel(xfad, yfad, zfad); 
  Sacado::Fad::DFad<ScalarT> wfad = w_vel(xfad, yfad, zfad); 
  ScalarT dvdz_ad = vfad.dx(1); 
  ScalarT dwdy_ad = wfad.dx(0); 
  ScalarT mu = mu_eval(x, y, z);
  ScalarT tau = mu*(dvdz_ad + dwdy_ad);   
  return tau; 
}

//returns expression for tau33 for MMF Test A ISMIP-HOM test case 
template<typename ScalarT>
ScalarT tau33(const ScalarT& x, const ScalarT& y, const ScalarT& z){
  int num_deriv = 1;  
  Sacado::Fad::DFad<ScalarT> xfad(x);
  Sacado::Fad::DFad<ScalarT> yfad(y); 
  Sacado::Fad::DFad<ScalarT> zfad(num_deriv, 0, z); 
  Sacado::Fad::DFad<ScalarT> wfad = w_vel(xfad, yfad, zfad); 
  ScalarT dwdz_ad = wfad.dx(0); 
  ScalarT mu = mu_eval(x, y, z);
  ScalarT p = press(x,y,z);  
  ScalarT tau = 2.0*mu*dwdz_ad - p;   
  return tau; 
}


template<typename EvalT, typename Traits>
StokesBodyForce<EvalT, Traits>::
StokesBodyForce(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  force(p.get<std::string>("Body Force Name"),dl->qp_vector),
  A(1.0), 
  n(3.0)
{

  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  A = bf_list->get("Glen's Law A", 1.0); 
  n = bf_list->get("Glen's Law n", 3.0); 
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "Gravity") {
    bf_type = GRAVITY;
  }
  else if (type == "Poly") {
    bf_type = POLY;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "PolySacado") {
    bf_type = POLYSACADO;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"), dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "SinSin") {
    bf_type = SINSIN;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "FullStokesBasal") {
    bf_type = FULLSTOKESBASAL;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "SinSinGlen") {
    bf_type = SINSINGLEN;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "SinCosZ") {
    bf_type = SINCOSZ;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }
  else if (type == "TestAMMF") {
    bf_type = TESTAMMF;  
    muFELIX = PHX::MDField<ScalarT,Cell,QuadPoint>(
            p.get<std::string>("FELIX Viscosity QP Variable Name"),dl->qp_scalar);
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(
            p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient);
    this->addDependentField(muFELIX); 
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  if (bf_type == GRAVITY) {
    if (bf_list->isType<Teuchos::Array<double> >("Gravity Vector"))
      gravity = bf_list->get<Teuchos::Array<double> >("Gravity Vector");
    else {
      gravity.resize(numDims);
      gravity[numDims-1] = -1.0;
    }
  }
  this->setName("StokesBodyForce"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (bf_type == GRAVITY) {
  }
  else if (bf_type == POLY || bf_type == SINSIN || bf_type == SINSINGLEN || bf_type == FULLSTOKESBASAL || bf_type == SINCOSZ || bf_type == POLYSACADO || bf_type == TESTAMMF) {
    this->utils.setFieldData(muFELIX,fm);
    this->utils.setFieldData(coordVec,fm);
  }

  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < numDims; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
 else if (bf_type == GRAVITY) {
 for (std::size_t cell=0; cell < workset.numCells; ++cell) 
   for (std::size_t qp=0; qp < numQPs; ++qp) {
     for (std::size_t i=0; i < numDims; ++i) { 
	 force(cell,qp,i) = -1.0*g*rho*gravity[i];
      }
   }
 }

 //The following is hard-coded for a 2D Stokes problem with manufactured solution
 else if (bf_type == POLY) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       //Irina Debug for Phalanx
       //ScalarT* f = &force(cell,qp,0);
       MeshScalarT X0 = coordVec(cell,qp,0);
       MeshScalarT X1 = coordVec(cell,qp,1);
       ScalarT muqp = muFELIX(cell,qp);
       force(cell,qp,0) =  40.0*muqp*(2.0*X1*X1 - 3.0*X1+1.0)*X1*(6.0*X0*X0 -6.0*X0 + 1.0)
              + 120*muqp*(X0-1.0)*(X0-1.0)*X0*X0*(2.0*X1-1.0) 
              + 10.0*muqp;      
       force(cell,qp,1) = - 120.0*muqp*(1.0-X1)*(1.0-X1)*X1*X1*(2.0*X0-1.0)
              - 40.0*muqp*(2.0*X0*X0 - 3.0*X0 + 1.0)*X0*(6.0*X1*X1 - 6.0*X1 + 1.0)
              - 5*muqp*X1;
     }
   }
 }
 // Doubly-periodic MMS derived by Irina. 
 else if (bf_type == SINSIN) {
   double xphase=0.0, yphase=0.0; // Expose as parameters for verification
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       //Irina Debug
       //ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       ScalarT muqp = muFELIX(cell,qp);

       force(cell,qp,0) = -4.0*muqp*pi*(2*pi-1)*sin(x2pi + xphase)*sin(y2pi + yphase);
       force(cell,qp,1) = -4.0*muqp*pi*(2*pi+1)*cos(x2pi + xphase)*cos(y2pi + yphase);
     }
   }
 }
 else if (bf_type == FULLSTOKESBASAL) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {   
       //Irina Debug   
      // ScalarT* f = &force(cell,qp,0);
       MeshScalarT x = coordVec(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       ScalarT muqp = muFELIX(cell,qp);
       force(cell,qp,0) = -2.0*pi*sin(y2pi)*(-1.0*muqp*exp(x) + cos(x2pi) + 4.0*muqp*pi*pi*exp(x));
       force(cell,qp,1) = -1.0*cos(y2pi)*(4.0*muqp*pi*pi*exp(x) - muqp*exp(x)  + 2.0*pi*sin(x2pi)); 
     }
   }
 }
 //MMS test case for 2D nonlinear Stokes with Glen's law on unit square 
 else if (bf_type == SINSINGLEN) {
   double xphase=0.0, yphase=0.0; 
   double r = 3*pi;
   //cout << "muqp force: " << endl; 
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       //Irina Debug
       //ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       //MeshScalarT muqp = x2pi*y2pi + r;
       MeshScalarT muargt = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r;  
       MeshScalarT muqp = 0.5*pow(A, -1.0/n)*pow(muargt, 1.0/n - 1.0);
       MeshScalarT dudx = 2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) + r; 
       MeshScalarT dudy = -2.0*pi*sin(x2pi + xphase)*cos(y2pi + yphase); 
       MeshScalarT dvdx = 2.0*pi*sin(x2pi + xphase)*sin(y2pi+yphase); 
       MeshScalarT dvdy = -2.0*pi*cos(x2pi + xphase)*cos(y2pi + yphase) - r;
       MeshScalarT dmuargtdx = -4.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase); 
       MeshScalarT dmuargtdy = -4.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase);  
       //cout.precision(10);
       //if (cell == 10)  
       //   cout << muqp << endl; 
       force(cell,qp,0) = -8.0*pi*pi*sin(x2pi + xphase)*cos(y2pi + yphase)*(muqp - 1.0) 
            + 0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*dudx + dmuargtdy*dudy);
       force(cell,qp,1) = 8.0*pi*pi*cos(x2pi + xphase)*sin(y2pi + yphase)*(muqp + 1.0) 
            + 0.5*pow(A, -1.0/n)*(1.0/n - 1.0)*pow(muargt, 1.0/n - 2.0)*(dmuargtdx*dvdx + dmuargtdy*dvdy);
     }
   }
 }
 else if (bf_type == POLYSACADO) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
      //Irina Debug
      //  ScalarT* f = &force(cell,qp,0);
       //MeshScalarT& x = coordVec(cell,qp,0);
       //MeshScalarT& y = coordVec(cell,qp,1);
       ScalarT X = coordVec(cell,qp,0);
       ScalarT Y = coordVec(cell,qp,1); 
       int num_deriv = 2; 
       Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, X); 
       Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, Y); 
       Sacado::Fad::DFad<ScalarT> tau11fad = tau112d(xfad, yfad); 
       Sacado::Fad::DFad<ScalarT> tau12fad = tau122d(xfad, yfad); 
       Sacado::Fad::DFad<ScalarT> tau22fad = tau222d(xfad, yfad); 
       force(cell,qp,0) = 1.0*(tau11fad.dx(0) + tau12fad.dx(1)); 
       force(cell,qp,1) = 1.0*(tau12fad.dx(0) + tau22fad.dx(1));
     }
   }
 }
 else if (bf_type == TESTAMMF) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       //Irina Debug
       //ScalarT* f = &force(cell,qp,0);
       //MeshScalarT& x = coordVec(cell,qp,0);
       //MeshScalarT& y = coordVec(cell,qp,1);
       //MeshScalarT& z = coordVec(cell,qp,2);
       ScalarT X = coordVec(cell,qp,0); 
       ScalarT Y = coordVec(cell,qp,1); 
       ScalarT Z = coordVec(cell,qp,3); 
       int num_deriv = 3; 
       Sacado::Fad::DFad<ScalarT> xfad(num_deriv, 0, X); 
       Sacado::Fad::DFad<ScalarT> yfad(num_deriv, 1, Y); 
       Sacado::Fad::DFad<ScalarT> zfad(num_deriv, 2, Z);
       Sacado::Fad::DFad<ScalarT> tau11fad = tau11(xfad, yfad, zfad); 
       Sacado::Fad::DFad<ScalarT> tau12fad = tau12(xfad, yfad, zfad); 
       Sacado::Fad::DFad<ScalarT> tau13fad = tau13(xfad, yfad, zfad); 
       Sacado::Fad::DFad<ScalarT> tau22fad = tau22(xfad, yfad, zfad); 
       Sacado::Fad::DFad<ScalarT> tau23fad = tau23(xfad, yfad, zfad); 
       Sacado::Fad::DFad<ScalarT> tau33fad = tau33(xfad, yfad, zfad); 
       force(cell,qp,0) = tau11fad.dx(0) + tau12fad.dx(1) + tau13fad.dx(2);  
       force(cell,qp,1) = tau12fad.dx(0) + tau22fad.dx(1) + tau23fad.dx(2); 
       force(cell,qp,2) = tau13fad.dx(0) + tau23fad.dx(1) + tau33fad.dx(2) + rho*g; 
     }
   }
 }
 // Doubly-periodic MMS with polynomial in Z
 else if (bf_type == SINCOSZ) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {      
       //Irina Debug
       //ScalarT* f = &force(cell,qp,0);
       MeshScalarT x2pi = 2.0*pi*coordVec(cell,qp,0);
       MeshScalarT y2pi = 2.0*pi*coordVec(cell,qp,1);
       //Irina Debug
       //MeshScalarT& z = coordVec(cell,qp,2);
       //ScalarT& muqp = muFELIX(cell,qp);
       MeshScalarT z = coordVec(cell,qp,2);
       ScalarT muqp = muFELIX(cell,qp);

       /*ScalarT t1 = muqp*z*(1-z)*(1-2*z);
       ScalarT t2 = muqp*(1-6*z+6*z*z);
       ScalarT t3 = muqp*z*z*(1-z)*(1-z);

       f[0] =  (-8*pi*pi*t1 + t2 + 4*pi*pi)*sin(x2pi)*sin(y2pi);
       f[1] = -(-8*pi*pi*t1 + t2 + 4*pi*pi)*cos(x2pi)*cos(y2pi);
       f[2] = -2*pi*(-8*pi*pi*t3 + 2*t1 + 1)*cos(x2pi)*sin(y2pi);
       */
       ScalarT t1 = muqp*z*(1.0-z)*(1.0-2.0*z); 
       ScalarT t2 = muqp*(12.0*z - 6.0); 
       ScalarT t3 = muqp*z*z*(1.0-z)*(1.0-z); 
       ScalarT t4 = muqp*(1.0-6.0*z+6.0*z*z); 
     
       force(cell,qp,0) = (-8.0*pi*pi*t1 + t2 + 4.0*pi*pi*z)*sin(x2pi)*sin(y2pi); 
       force(cell,qp,1) = (8.0*pi*pi*t1 - t2 - 4.0*pi*pi*z)*cos(x2pi)*cos(y2pi); 
       force(cell,qp,2) = (16.0*pi*pi*pi*t3 - 4.0*pi*t4 - 2.0*pi)*cos(x2pi)*sin(y2pi); 
     }
   }
 }
}

}

