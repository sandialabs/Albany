//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//
// To do:
//  1.  Expand to lowest symmetry group (See Nye).
//  3.  Initialize electric displacement from (x,E), i.e., partially invert
//

#define THREE_D 3

#include <Intrepid2_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM
{

/******************************************************************************/
template<typename EvalT, typename Traits>
FerroicModel<EvalT, Traits>::
FerroicModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
/******************************************************************************/
{

  TEUCHOS_TEST_FOR_EXCEPTION(num_dims_ != THREE_D, std::invalid_argument,
    ">>> ERROR (FerroicModel): Only valid for 3D.");

  // PARSE MATERIAL BASIS
  //
  R.set_dimension(THREE_D); R.clear();
  if(p->isType<Teuchos::ParameterList>("Material Basis")){
    const Teuchos::ParameterList& pBasis = p->get<Teuchos::ParameterList>("Material Basis");
    LCM::parseBasis(pBasis,R);
  } else {
    R(0,0) = 1.0; R(1,1) = 1.0; R(2,2) = 1.0;
  }

  if(p->isType<Teuchos::Array<RealType>>("Bin Fractions") )
    initialBinFractions = p->get<Teuchos::Array<RealType>>("Bin Fractions");
  else 
    initialBinFractions.resize(0);

  // Parse phases
  //
  int nphases = p->get<int>("Number of Phases");
  for(int i=0; i<nphases; i++){
    Teuchos::ParameterList& pParam = p->get<Teuchos::ParameterList>(Albany::strint("Phase",i+1));
    crystalPhases.push_back(Teuchos::rcp(new CrystalPhase(R, pParam)));
  }

  // Parse variants
  //
  if(initialBinFractions.size() > 0){
    Teuchos::ParameterList& vParams = p->get<Teuchos::ParameterList>("Variants");
    int nvars = vParams.get<int>("Number of Variants");
    TEUCHOS_TEST_FOR_EXCEPTION(initialBinFractions.size() != nvars, std::invalid_argument,
       ">>> ERROR (FerroicModel): 'Number of Variants' must equal length of 'Bin Fractions' array");
    for(int i=0; i<nvars; i++){
      Teuchos::ParameterList& vParam = vParams.get<Teuchos::ParameterList>(Albany::strint("Variant",i+1));
      crystalVariants.push_back(Teuchos::rcp(new CrystalVariant(crystalPhases, vParam)));
    }
  } else {
    // no variants specified.  Create single dummy variant.
    initialBinFractions.resize(1);
    initialBinFractions[0] = 1.0;
  }

  //C.set_dimension(R.get_dimension());
  //h.set_dimension(R.get_dimension());
  //beta.set_dimension(R.get_dimension());

  // create transitions
  //
  int nVariants = crystalVariants.size();
  for(int I=0; I<nVariants; I++)
    for(int J=0; J<nVariants; J++)
      transitions.push_back(Teuchos::rcp(new Transition(crystalVariants[I],crystalVariants[J])));

  // create/initialize transition matrix
  //
  int nTransitions = transitions.size();
  aMatrix.resize(nVariants, nTransitions);
  for(int I=0; I<nVariants; I++){
    for(int J=0; J<nVariants; J++){
      aMatrix(I,nVariants*I+J) = -1.0;
      aMatrix(J,nVariants*I+J) = 1.0;
    }
    aMatrix(I,nVariants*I+I) = 0.0;
  }

  tBarrier.resize(transitions.size());

  // parse critical energies
  // 
  if(p->isType<Teuchos::ParameterList>("Critical Values")){
    const Teuchos::ParameterList& cParams = p->get<Teuchos::ParameterList>("Critical Values");
    int transitionIndex = 0;
    for(int i=0; i<nVariants; i++){
      Teuchos::Array<RealType> array = cParams.get<Teuchos::Array<RealType>>(Albany::strint("Variant",i+1));
      TEUCHOS_TEST_FOR_EXCEPTION(array.size()!=nVariants, std::invalid_argument,
         ">>> ERROR (FerroicModel): List of critical values for variant " << i+1 << " is wrong length");
      for(int j=0; j<nVariants; j++){
        tBarrier[transitionIndex] = array[j];
        transitionIndex++;
      }
    }
    alphaParam = cParams.get<RealType>("alpha");
    gammaParam = cParams.get<RealType>("gamma");
  }
 

  // DEFINE THE EVALUATED FIELDS
  //
  stressName = "Stress";
  this->eval_field_map_.insert(std::make_pair(stressName, dl->qp_tensor));

  edispName = "Electric Displacement";
  this->eval_field_map_.insert(std::make_pair(edispName, dl->qp_vector));

  // DEFINE THE DEPENDENT FIELDS
  //
  strainName = "Strain";
  this->dep_field_map_.insert(std::make_pair(strainName, dl->qp_tensor));

  efieldName = "Electric Potential Gradient";
  this->dep_field_map_.insert(std::make_pair(efieldName, dl->qp_vector));

  // DEFINE STATE VARIABLES (output)
  //
  for(int i=0; i<nVariants; i++){
    std::string binName = Albany::strint("Bin Fraction",i+1);
    binNames.push_back(binName);
    this->eval_field_map_.insert(std::make_pair(binName, dl->qp_scalar));
  }
  
  // bin fractions
  for(int i=0; i<nVariants; i++){
    this->num_state_variables_++;
    this->state_var_names_.push_back(binNames[i]);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(initialBinFractions[i]);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);
  }
  
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(stressName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // strain
  this->num_state_variables_++;
  this->state_var_names_.push_back(strainName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  
  // edisp
  this->num_state_variables_++;
  this->state_var_names_.push_back(edispName);
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // efield
  this->num_state_variables_++;
  this->state_var_names_.push_back(efieldName);
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  
}

/******************************************************************************/
template<typename EvalT, typename Traits>
void FerroicModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
/******************************************************************************/
{
  PHX::MDField<ScalarT> strain = *dep_fields[strainName];
  PHX::MDField<ScalarT> Gradp = *dep_fields[efieldName];

  PHX::MDField<ScalarT> stress = *eval_fields[stressName];
  PHX::MDField<ScalarT> edisp  = *eval_fields[edispName];


  int nVariants = crystalVariants.size();
  Teuchos::Array<PHX::MDField<ScalarT>> newBinFractions(nVariants);
  Teuchos::Array<Albany::MDArray> oldBinFractions(nVariants);
  for(int i=0; i<nVariants; i++){
    oldBinFractions[i] = (*workset.stateArrayPtr)[binNames[i] + "_old"];
    newBinFractions[i] = *eval_fields[binNames[i]];
  }

  int numCells = workset.numCells;

  Intrepid2::Tensor<ScalarT> X(THREE_D), x(THREE_D), linear_x(THREE_D);
  Intrepid2::Vector<ScalarT> E(THREE_D), D(THREE_D), linear_D(THREE_D);
  Teuchos::Array<ScalarT> oldfractions(nVariants);
  Teuchos::Array<ScalarT> newfractions(nVariants);

  Intrepid2::FieldContainer<int> transitionMap;
  transitionMap.resize(transitions.size());

  for (int cell=0; cell < numCells; ++cell) {
    for (int qp=0; qp < num_pts_; ++qp) {

      x.fill(strain,cell,qp,0,0);
      E.fill(Gradp,cell,qp,0);
      E *= -1.0;

      for(int vnt=0; vnt<nVariants; vnt++){
        oldfractions[vnt] = oldBinFractions[vnt](cell,qp);
        newfractions[vnt] = newBinFractions[vnt](cell,qp);
      }

      // Evaluate trial state 
      //
      computeState(oldfractions, x,X,linear_x, E,D,linear_D);
  
      // Find active dissipation mechanisms at trial state
      //
      findActiveTransitions(transitionMap, oldfractions, X,linear_x, E,linear_D);

      // If active, update volume fractions
      //
      findEquilibriumState(transitionMap, oldfractions, newfractions, x, E);

      // Evaluate new state at updated bin fractions
      //
      computeState(newfractions, x,X,linear_x, E,D,linear_D);

      for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
          stress(cell,qp,i,j) = X(i,j);
      for(int i=0; i<3; i++)
        edisp(cell,qp,i) = D(i);

    }
  }

  
}

/******************************************************************************/
template<typename EvalT, typename Traits>
void FerroicModel<EvalT, Traits>::findEquilibriumState(
Intrepid2::FieldContainer<int>& transitionMap,
Teuchos::Array<ScalarT>& oldfractions,
Teuchos::Array<ScalarT>& newfractions,
Intrepid2::Tensor<ScalarT>& x, Intrepid2::Vector<ScalarT>& E)
/******************************************************************************/
{

  int nActive=0;
  int nVariants = oldfractions.size();
  int nTransitions = transitionMap.size();
  for(int i=0; i<nTransitions; i++)
    if(transitionMap[i] >= 0) nActive++;

  int totalDofs = nActive+nVariants;
  std::vector<ScalarT> S(totalDofs);
  std::vector<ScalarT> R(totalDofs);
  std::vector<ScalarT> dRdS(totalDofs*totalDofs);

  

  for(int i=0; i<nActive; i++) S[i] = 0.0;

  LocalNonlinearSolver<EvalT, Traits> solver;

  int iter=0;
  ScalarT initNorm = 0.0;
  while(nActive){

    computeResidualandJacobian(transitionMap, x, E,
                               oldfractions, S, R, dRdS);
                               
    if( converged(R, iter, initNorm) ) break;
  
    // dRdS has lower limit equality constraints.
    std::vector<ScalarT> Scopy(S);
    std::vector<ScalarT> Rcopy(R);
    std::vector<ScalarT> dRdScopy(dRdS);
    solver.solve(dRdScopy, Scopy, Rcopy);

    // check for negative KKT multipliers.  If present, constrain them to zero
    // and recompute increment
    bool resolve = false;
    for(int i=nActive; i<totalDofs; i++){
      if(Scopy[i] < 0.0){
        resolve = true;
        for(int j=0; j<totalDofs; j++) dRdS[i + totalDofs*j] = 0.0;
        dRdS[i + totalDofs*i] = 1.0;
        R[i] = 0.0;
      }
    }
    if(resolve) solver.solve(dRdS, S, R);

    iter++;
  }

  // use unconstrained dRdS to compute Fad Info.

  if(nActive) solver.computeFadInfo(dRdS, S, R);

  // compute updated fractions
  //
  for(int I=0;I<nVariants;I++){
    newfractions[I] = oldfractions[I];
    for(int i=0;i<nTransitions;i++){
      if(transitionMap[i] >= 0){
        newfractions[I] += S[transitionMap[i]]*aMatrix(I,i);
      }
    }
  }
 

}


/******************************************************************************/
template<typename EvalT, typename Traits>
bool FerroicModel<EvalT, Traits>::converged(std::vector<ScalarT>& R, 
                                            int iteration, ScalarT& initNorm)
/******************************************************************************/
{
 
  bool converged = true;

  int nvals = R.size();
  ScalarT norm = 0.0;
  for(int ival=0; ival<nvals; ival++){
    norm += R[ival]*R[ival];
  }
  if(norm > 0.0)
   norm = sqrt(norm);

  if(iteration==0) initNorm = norm;
   
  if(initNorm == 0.0) 
    converged = true;
  else
    converged = (norm/initNorm < 1.0e-11 || norm < 1.0e-11 || iteration > 20);

  return converged;
}

/******************************************************************************/
template<typename EvalT, typename Traits>
void FerroicModel<EvalT, Traits>::
computeResidualandJacobian(
  Intrepid2::FieldContainer<int> transitionMap,
  Intrepid2::Tensor<ScalarT>& x, Intrepid2::Vector<ScalarT>& E,
  Teuchos::Array<ScalarT>& fractions,
  std::vector<ScalarT>& S, std::vector<ScalarT>& R, std::vector<ScalarT>& dRdS)
/******************************************************************************/
{

  Intrepid2::Tensor<DFadType> X; X.set_dimension(THREE_D); X.clear();
  Intrepid2::Tensor<DFadType> linear_x; linear_x.set_dimension(THREE_D); linear_x.clear();
  Intrepid2::Vector<DFadType> D; D.set_dimension(THREE_D); D.clear();
  Intrepid2::Vector<DFadType> linear_D; linear_D.set_dimension(THREE_D); linear_D.clear();

  int nTotalDofs = S.size();
  int nVariants = crystalVariants.size();
  int nActive = nTotalDofs - nVariants;
  int nTransitions = transitionMap.size();

  Teuchos::Array<DFadType> fractionsNew(nVariants);
  Teuchos::Array<DFadType> Rfad(nActive);
  Teuchos::Array<DFadType> Sfad(nActive);
  Teuchos::Array<ScalarT> Sval(nActive);

  for(std::size_t i=0; i<nActive; i++){
    Sval[i] = Sacado::ScalarValue<ScalarT>::eval(S[i]);
    Sfad[i] = DFadType(nActive, i, Sval[i]);
  }

  // compute updated fractions
  //
  for(int I=0;I<nVariants;I++){
    fractionsNew[I] = fractions[I];
    for(int i=0;i<nTransitions;i++){
      if(transitionMap[i] >= 0){
        fractionsNew[I] += Sfad[transitionMap[i]]*aMatrix(I,i);
      }
    }
  }

  computeState(fractionsNew,
               x, X, linear_x, 
               E, D, linear_D);

  // compute residual
  //
  ScalarT half = 1.0/2.0;
  for(int I=0;I<nVariants;I++){
    DFadType fracI = fractionsNew[I];
    for(int J=0;J<nVariants;J++){
      int i=I*nVariants+J;
      if(transitionMap[i] >= 0){
        Transition& transition = *transitions[i];
        int lindex = transitionMap[i];
        Rfad[lindex] = -tBarrier[i]*(1.0+alphaParam*fracI+gammaParam*fracI*fracI)
                       -dotdot(transition.transStrain, X) - dot(transition.transEDisp, E);
      }
    }
  }
  for(int I=0;I<nVariants;I++){
    DFadType myRate(0.0);
    CrystalVariant& variant = *crystalVariants[I];
    myRate += dotdot(linear_x,dotdot(variant.C,linear_x)-dot(linear_D,variant.h))*half;
    myRate += dot(linear_D,dot(variant.beta,linear_D)-dotdot(variant.h,linear_x))*half;
    for(int i=0;i<nTransitions;i++){
      if(transitionMap[i] >= 0){
        RealType aVal = aMatrix(I,i);
        if( aVal == 0.0 ) continue;
        int lindex = transitionMap[i];
        Rfad[lindex] += aVal*myRate;
      }
    }
  }
 
  // load residual
  //
  for(int i=0; i<nActive; i++)
    R[i] = Rfad[i].val();

  // add lower bound constraint
  //
  for(int I=0; I<nVariants; I++){
    R[nActive+I] = fractions[I];
  }

  // load Jacobian
  //
  for(int i=0; i<nActive; i++)
    for(int j=0; j<nActive; j++)
      dRdS[i + nTotalDofs*j] = Rfad[i].dx(j);

  // add lower bound constraint
  //  
  for(int I=0; I<nVariants; I++){
    bool activeVariant = false;
    for(int i=0;i<nTransitions;i++){
      if(transitionMap[i] >= 0){
        int lindex = transitionMap[i];
        dRdS[nActive+I + nTotalDofs*lindex] = -aMatrix(I,i);
        dRdS[lindex + nTotalDofs*(I+nActive)] = aMatrix(I,i);
        if(aMatrix(I,i) != 0.0) activeVariant = true;
      }
    }
    if(!activeVariant){
     dRdS[nActive+I + nTotalDofs*(nActive+I)] = 1.0;
     R[nActive+I] = 0.0;
    }
  }


  // diagonal shifting to avoid conditioning problems
//  ScalarT alpha=0.0;
//  for(int i=0; i<nActive; i++){
//    alpha += dRdS[i+nActive*i];
//  }
//  alpha /= nActive;
//  alpha *= 1e-6;
//  for(int i=0; i<nTotalDofs; i++)
//    dRdS[i + nTotalDofs*i] += alpha;
//  
}

/******************************************************************************/
template<typename EvalT, typename Traits>
template<typename T>
void FerroicModel<EvalT, Traits>::
computeState(Teuchos::Array<T>& fractions,
             Intrepid2::Tensor<ScalarT>& x, Intrepid2::Tensor<T>& X, Intrepid2::Tensor<T>& linear_x,
             Intrepid2::Vector<ScalarT>& E, Intrepid2::Vector<T>& D, Intrepid2::Vector<T>& linear_D)
/******************************************************************************/
{
  Intrepid2::Tensor4<T> C; 
  C.set_dimension(THREE_D); C.clear();

  Intrepid2::Tensor3<T> h; 
  h.set_dimension(THREE_D); h.clear();

  Intrepid2::Tensor<T> beta; 
  beta.set_dimension(THREE_D); beta.clear();

  Intrepid2::Tensor<T> remanent_x; 
  remanent_x.set_dimension(THREE_D); remanent_x.clear();

  Intrepid2::Vector<T> remanent_D; 
  remanent_D.set_dimension(THREE_D); remanent_D.clear();

  int nVariants = crystalVariants.size();
  for(int i=0; i<nVariants; i++){
    const CrystalVariant& variant = *crystalVariants[i];
    remanent_x += fractions[i]*variant.spontStrain;
    remanent_D += fractions[i]*variant.spontEDisp;
    C += fractions[i]*variant.C;
    h += fractions[i]*variant.h;
    beta += fractions[i]*variant.beta;
  }
  Intrepid2::Tensor<T> eps = Intrepid2::inverse(beta);

  linear_x = x - remanent_x;

  linear_D = dot( eps, E+dotdot(h,linear_x) );
  X = dotdot(C,linear_x) - dot(linear_D,h);

  D = linear_D + remanent_D;

}

/******************************************************************************/
template<typename EvalT, typename Traits>
void FerroicModel<EvalT, Traits>::findActiveTransitions(
Intrepid2::FieldContainer<int>& transitionMap,
Teuchos::Array<ScalarT>& fractions,
Intrepid2::Tensor<ScalarT>& X, Intrepid2::Tensor<ScalarT>& linear_x,
Intrepid2::Vector<ScalarT>& E, Intrepid2::Vector<ScalarT>& linear_D)
/******************************************************************************/
{

  int nTransitions = transitions.size();
  int nVariants = initialBinFractions.size();
  Teuchos::Array<ScalarT> tForce(nTransitions);
//  for(int i=0;i<nTransitions;i++){
//
//
//    ScalarT myRate = 0.0;
//
//    for(int I=0;I<nVariants;I++){
//      RealType aVal = aMatrix(I,i);
//      if( aVal == 0.0 ) continue;
//      CrystalVariant& variant = *crystalVariants[I];
//      myRate += dotdot(linear_x,dotdot(variant.C,linear_x)-dot(linear_D,variant.h))*aVal/2.0;
//      myRate += dot(linear_D,dot(variant.beta,linear_D)-dotdot(variant.h,linear_x))*aVal/2.0;
//    }
//    Transition& transition = *transitions[i];
//    myRate -= dotdot(transition.transStrain, X) + dot(transition.transEDisp, E);
//
//    tForce(i) = myRate - tBarrier(i);   
//  }

// is this equivalent to the commented section above?
  for(int I=0;I<nVariants;I++){
    ScalarT fracI = fractions[I];
    for(int J=0;J<nVariants;J++){
      int i=I*nVariants+J;
      Transition& transition = *transitions[i];
      tForce[i] = -tBarrier[i]*(1.0+alphaParam*fracI+gammaParam*fracI*fracI) 
                  -dotdot(transition.transStrain, X) + dot(transition.transEDisp, E);
    }
  }
  for(int I=0;I<nVariants;I++){
    ScalarT myRate = 0.0;
    CrystalVariant& variant = *crystalVariants[I];
    myRate += dotdot(linear_x,dotdot(variant.C,linear_x)-dot(linear_D,variant.h))/2.0;
    myRate += dot(linear_D,dot(variant.beta,linear_D)-dotdot(variant.h,linear_x))/2.0;
    for(int i=0;i<nTransitions;i++){
      RealType aVal = aMatrix(I,i);
      if( aVal == 0.0 ) continue;
      tForce[i] += aVal*myRate;
    }
  }

  int transition=0, nActive=0;
  transitionMap.initialize(-1);
  for(int I=0;I<nVariants;I++){
    if(fractions[I] <= 0.0) continue;
    for(int J=0;J<nVariants;J++){
      transition = I*nVariants+J;
      if(tForce[transition] < -0.01){
        transitionMap[transition] = nActive;
        nActive++;
      }
    }
  }

}

/******************************************************************************/
template<typename EvalT, typename Traits>
void FerroicModel<EvalT, Traits>::
computeStateParallel(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
     ">>> ERROR (FerroicModel): computeStateParallel not implemented");
}

/******************************************************************************/
void changeBasis(Intrepid2::Tensor4<RealType>& inMatlBasis, 
            const Intrepid2::Tensor4<RealType>& inGlobalBasis,
            const Intrepid2::Tensor<RealType>& R)
/******************************************************************************/
{
    int num_dims = R.get_dimension();
    inMatlBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int j=0; j<num_dims; j++)
      for(int k=0; k<num_dims; k++)
       for(int l=0; l<num_dims; l++)
        for(int q=0; q<num_dims; q++)
         for(int r=0; r<num_dims; r++)
          for(int s=0; s<num_dims; s++)
           for(int t=0; t<num_dims; t++)
            inMatlBasis(i,j,k,l) += inGlobalBasis(q,r,s,t)*R(i,q)*R(j,r)*R(k,s)*R(l,t);
}
/******************************************************************************/
void changeBasis(Intrepid2::Tensor3<RealType>& inMatlBasis, 
            const Intrepid2::Tensor3<RealType>& inGlobalBasis,
            const Intrepid2::Tensor<RealType>& R)
/******************************************************************************/
{
    int num_dims = R.get_dimension();
    inMatlBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int j=0; j<num_dims; j++)
      for(int k=0; k<num_dims; k++)
       for(int q=0; q<num_dims; q++)
        for(int r=0; r<num_dims; r++)
         for(int s=0; s<num_dims; s++)
           inMatlBasis(i,j,k) += inGlobalBasis(q,r,s)*R(i,q)*R(j,r)*R(k,s);
}
/******************************************************************************/
void changeBasis(Intrepid2::Tensor<RealType>& inMatlBasis, 
            const Intrepid2::Tensor<RealType>& inGlobalBasis,
            const Intrepid2::Tensor<RealType>& R)
/******************************************************************************/
{
    int num_dims = R.get_dimension();
    inMatlBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int j=0; j<num_dims; j++)
      for(int q=0; q<num_dims; q++)
       for(int r=0; r<num_dims; r++)
        inMatlBasis(i,j) += inGlobalBasis(q,r)*R(i,q)*R(j,r);
}
/******************************************************************************/
void parseBasis(const Teuchos::ParameterList& pBasis,
           Intrepid2::Tensor<RealType>& R)
/******************************************************************************/
{
  if(pBasis.isType<Teuchos::Array<RealType>>("X axis")){
    Teuchos::Array<RealType> Xhat = pBasis.get<Teuchos::Array<RealType>>("X axis");
    R(0,0) = Xhat[0]; R(0,1) = Xhat[1]; R(0,2) = Xhat[2];
//    R(0,0) = Xhat[0]; R(1,0) = Xhat[1]; R(2,0) = Xhat[2];
  }
  if(pBasis.isType<Teuchos::Array<RealType>>("Y axis")){
    Teuchos::Array<RealType> Yhat = pBasis.get<Teuchos::Array<RealType>>("Y axis");
    R(1,0) = Yhat[0]; R(1,1) = Yhat[1]; R(1,2) = Yhat[2];
//    R(0,1) = Yhat[0]; R(1,1) = Yhat[1]; R(2,1) = Yhat[2];
  }
  if(pBasis.isType<Teuchos::Array<RealType>>("Z axis")){
    Teuchos::Array<RealType> Zhat = pBasis.get<Teuchos::Array<RealType>>("Z axis");
    R(2,0) = Zhat[0]; R(2,1) = Zhat[1]; R(2,2) = Zhat[2];
//    R(0,2) = Zhat[0]; R(1,2) = Zhat[1]; R(2,2) = Zhat[2];
  }
}

/******************************************************************************/
template<typename EvalT, typename Traits>
FerroicModel<EvalT, Traits>::
Transition::
Transition(Teuchos::RCP<CrystalVariant> from_, 
           Teuchos::RCP<CrystalVariant> to_) :
           fromVariant(from_), toVariant(to_)
/******************************************************************************/
{
  transStrain.set_dimension(THREE_D);
  transEDisp.set_dimension(THREE_D);

  transStrain = toVariant->spontStrain - fromVariant->spontStrain;
  transEDisp  = toVariant->spontEDisp  - fromVariant->spontEDisp;
}

/******************************************************************************/
template<typename EvalT, typename Traits>
FerroicModel<EvalT, Traits>::
CrystalVariant::CrystalVariant(Teuchos::Array<Teuchos::RCP<CrystalPhase>>& phases, 
                               Teuchos::ParameterList& vParam)
/******************************************************************************/
{

  TEUCHOS_TEST_FOR_EXCEPTION(phases.size()==0, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalVariant constructor passed empty list of Phases.");
  
  int phaseIndex;
  if(vParam.isType<int>("Phase")){
    phaseIndex = vParam.get<int>("Phase") ;
    phaseIndex--; // Ids are one-based.  Indices are zero-based.
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a phase.");

  TEUCHOS_TEST_FOR_EXCEPTION(phaseIndex < 0 || phaseIndex >= phases.size(), 
    std::invalid_argument,
    ">>> ERROR (FerroicModel): Requested phase has not been defined.");


  if(vParam.isType<Teuchos::ParameterList>("Crystallographic Basis")){
    R.set_dimension(phases[phaseIndex]->C.get_dimension());
    const Teuchos::ParameterList& 
    pBasis = vParam.get<Teuchos::ParameterList>("Crystallographic Basis");
    LCM::parseBasis(pBasis,R);
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a crystallograph basis.");

  if(vParam.isType<Teuchos::Array<RealType>>("Spontaneous Polarization")){
    Teuchos::Array<RealType> 
      inVals = vParam.get<Teuchos::Array<RealType>>("Spontaneous Polarization");
      TEUCHOS_TEST_FOR_EXCEPTION(inVals.size() != THREE_D, std::invalid_argument,
      ">>> ERROR (FerroicModel): Expected 3 terms 'Spontaneous Polarization' vector.");
      spontEDisp.set_dimension(THREE_D);
      for(int i=0; i<THREE_D; i++) spontEDisp(i) = inVals[i];
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous Polarization'.");

  if(vParam.isType<Teuchos::Array<RealType>>("Spontaneous Strain")){
    Teuchos::Array<RealType> 
      inVals = vParam.get<Teuchos::Array<RealType>>("Spontaneous Strain");
      TEUCHOS_TEST_FOR_EXCEPTION(inVals.size() != 6, std::invalid_argument,
      ">>> ERROR (FerroicModel): Expected 6 voigt terms 'Spontaneous Strain' tensor.");
      spontStrain.set_dimension(THREE_D);
      spontStrain(0,0) = inVals[0];
      spontStrain(1,1) = inVals[1];
      spontStrain(2,2) = inVals[2];
      spontStrain(1,2) = inVals[3];
      spontStrain(0,2) = inVals[4];
      spontStrain(0,1) = inVals[5];
      spontStrain(2,1) = inVals[3];
      spontStrain(2,0) = inVals[4];
      spontStrain(1,0) = inVals[5];
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous Strain'.");


  C.set_dimension(phases[phaseIndex]->C.get_dimension()); C.clear();
  LCM::changeBasis(C, phases[phaseIndex]->C, R);
  
  h.set_dimension(phases[phaseIndex]->h.get_dimension()); h.clear();
  LCM::changeBasis(h, phases[phaseIndex]->h, R);
  
  beta.set_dimension(phases[phaseIndex]->beta.get_dimension()); beta.clear();
  LCM::changeBasis(beta, phases[phaseIndex]->beta, R);
  
} 

/******************************************************************************/
template<typename EvalT, typename Traits>
FerroicModel<EvalT, Traits>::
CrystalPhase::CrystalPhase(Intrepid2::Tensor<RealType>& R,
                           Teuchos::ParameterList& cParam)
/******************************************************************************/
{
  // parse 
  //
  C11 = cParam.get<RealType>("C11");
  C33 = cParam.get<RealType>("C33");
  C12 = cParam.get<RealType>("C12");
  C23 = cParam.get<RealType>("C23");
  C44 = cParam.get<RealType>("C44");
  C66 = cParam.get<RealType>("C66");

  h31 = cParam.get<RealType>("h31");
  h33 = cParam.get<RealType>("h33");
  h15 = cParam.get<RealType>("h15");

  E11 = cParam.get<RealType>("Eps11");
  E33 = cParam.get<RealType>("Eps33");

  int num_dims = R.get_dimension();
  TEUCHOS_TEST_FOR_EXCEPTION(num_dims!=THREE_D, std::invalid_argument,
       ">>> ERROR (FerroicModel):  Only 3D supported");
  C.set_dimension(num_dims); C.clear();
  Intrepid2::Tensor4<RealType> Ctmp(num_dims); Ctmp.clear();
  h.set_dimension(num_dims); h.clear();
  Intrepid2::Tensor3<RealType> htmp(num_dims); htmp.clear();
  beta.set_dimension(num_dims); beta.clear();
  Intrepid2::Tensor<RealType> betatmp(num_dims); betatmp.clear();

  // create constants in tensor form
  Ctmp(0,0,0,0) = C11; Ctmp(0,0,1,1) = C12; Ctmp(0,0,2,2) = C23;
  Ctmp(1,1,0,0) = C12; Ctmp(1,1,1,1) = C11; Ctmp(1,1,2,2) = C23;
  Ctmp(2,2,0,0) = C23; Ctmp(2,2,1,1) = C23; Ctmp(2,2,2,2) = C33;
  Ctmp(0,1,0,1) = C66/2.0; Ctmp(1,0,1,0) = C66/2.0;
  Ctmp(0,2,0,2) = C44/2.0; Ctmp(2,0,2,0) = C44/2.0;
  Ctmp(1,2,1,2) = C44/2.0; Ctmp(2,1,2,1) = C44/2.0;

  htmp(0,0,2) = h15/2.0; htmp(0,2,0) = h15/2.0;
  htmp(1,1,2) = h15/2.0; htmp(1,2,1) = h15/2.0;
  htmp(2,0,0) = h31; htmp(2,1,1) = h31; htmp(2,2,2) = h33;

  Intrepid2::Tensor<RealType> eps(num_dims);
  eps.clear();

  eps(0,0) = E11;
  eps(1,1) = E11;
  eps(2,2) = E33;

  betatmp = Intrepid2::inverse(eps);

  // rotate into material basis
  LCM::changeBasis(C, Ctmp, R);
  LCM::changeBasis(h, htmp, R);
  LCM::changeBasis(beta, betatmp, R);
  
} 

}
