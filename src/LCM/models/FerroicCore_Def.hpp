//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/******************************************************************************/
template <typename EvalT, minitensor::Index M>
FM::DomainSwitching<EvalT, M>::DomainSwitching(
    Teuchos::Array<FM::CrystalVariant> const& crystalVariants,
    Teuchos::Array<FM::Transition> const&     transitions,
    Teuchos::Array<RealType> const&           transBarriers,
    Teuchos::Array<RealType> const&           binFractions,
    Kokkos::DynRankView<RealType> const&      aMatrix,
    minitensor::Tensor<ArgT, THREE_D> const&  x,
    minitensor::Vector<ArgT, THREE_D> const&  E,
    RealType                                  dt)
    : m_crystalVariants(crystalVariants),
      m_transitions(transitions),
      m_transBarriers(transBarriers),
      m_binFractions(binFractions),
      m_aMatrix(aMatrix),
      m_x(x),
      m_dt(dt)
/******************************************************************************/
{
  // compute trial state
  //
  minitensor::Tensor<ArgT, FM::THREE_D> X, linear_x;
  minitensor::Vector<ArgT, FM::THREE_D> linear_D;
  FM::computeInitialState(
      m_binFractions, m_crystalVariants, m_x, X, linear_x, E, m_D, linear_D);

  // set all transitions active for first residual eval
  //
  int nTransitions = m_transitions.size();
  m_transitionMap.resize(nTransitions);
  for (int J = 0; J < nTransitions; J++) { m_transitionMap[J] = J; }

  // evaluate residual at current bin fractions
  //
  minitensor::Vector<ArgT, FM::MAX_TRNS> zero;
  zero.set_dimension(nTransitions);
  zero.clear();
  minitensor::Vector<ArgT, FM::MAX_TRNS> residual = this->gradient(zero);

  // find active transitions
  //
  int nVariants  = m_binFractions.size();
  int transition = 0, nActive = 0;
  for (int J = 0; J < nTransitions; J++) m_transitionMap[J] = -1;
  for (int I = 0; I < nVariants; I++) {
    if (m_binFractions[I] <= 1.0e-10) continue;
    for (int J = 0; J < nVariants; J++) {
      transition = I * nVariants + J;
      if (residual[transition] < -0.01) {
        m_transitionMap[transition] = nActive;
        nActive++;
      }
    }
  }
  m_numActiveTransitions = nActive;
}

/******************************************************************************/
template <typename EvalT, minitensor::Index M>
template <typename T, minitensor::Index N>
T
FM::DomainSwitching<EvalT, M>::value(minitensor::Vector<T, N> const& x)
/******************************************************************************/
{
  return Base::value(*this, x);
}

/******************************************************************************/
template <typename EvalT, minitensor::Index M>
template <typename T, minitensor::Index N>
minitensor::Vector<T, N>
FM::DomainSwitching<EvalT, M>::gradient(
    minitensor::Vector<T, N> const& xi) const
/******************************************************************************/
{
  minitensor::Tensor<T, FM::THREE_D> X;
  X.clear();
  minitensor::Tensor<T, FM::THREE_D> linear_x;
  linear_x.clear();

  minitensor::Vector<T, FM::THREE_D> E;
  E.clear();
  minitensor::Vector<T, FM::THREE_D> linear_D;
  linear_D.clear();

  // apply transition increment
  //
  Teuchos::Array<T> fractionsNew(m_binFractions.size());
  computeBinFractions(
      xi, fractionsNew, m_binFractions, m_transitionMap, m_aMatrix);

  minitensor::Tensor<T, FM::THREE_D> const x_peeled =
      LCM::peel_tensor<EvalT, T, N, FM::THREE_D>()(m_x);

  minitensor::Vector<T, FM::THREE_D> const D_peeled =
      LCM::peel_vector<EvalT, T, N, FM::THREE_D>()(m_D);

  // compute new state
  //
  computeRelaxedState(
      fractionsNew,
      m_crystalVariants,
      x_peeled,
      X,
      linear_x,
      E,
      D_peeled,
      linear_D);

  // compute new residual
  //
  auto const               num_unknowns = xi.get_dimension();
  minitensor::Vector<T, N> residual(num_unknowns);
  computeResidual(
      residual,
      fractionsNew,
      m_transitionMap,
      m_transitions,
      m_crystalVariants,
      m_transBarriers,
      m_aMatrix,
      X,
      linear_x,
      E,
      linear_D);

  return residual;
}

/******************************************************************************/
template <typename EvalT, minitensor::Index M>
template <typename T, minitensor::Index N>
minitensor::Tensor<T, N>
FM::DomainSwitching<EvalT, M>::hessian(minitensor::Vector<T, N> const& xi)
/******************************************************************************/
{
  return Base::hessian(*this, xi);
}

/******************************************************************************/
template <typename DataT>
void
FM::changeBasis(
    minitensor::Tensor4<DataT, FM::THREE_D>&       inMatlBasis,
    const minitensor::Tensor4<DataT, FM::THREE_D>& inGlobalBasis,
    const minitensor::Tensor<DataT, FM::THREE_D>&  R)
/******************************************************************************/
{
  int num_dims = R.get_dimension();
  inMatlBasis.clear();
  for (int i = 0; i < num_dims; i++)
    for (int j = 0; j < num_dims; j++)
      for (int k = 0; k < num_dims; k++)
        for (int l = 0; l < num_dims; l++)
          for (int q = 0; q < num_dims; q++)
            for (int r = 0; r < num_dims; r++)
              for (int s = 0; s < num_dims; s++)
                for (int t = 0; t < num_dims; t++)
                  inMatlBasis(i, j, k, l) += inGlobalBasis(q, r, s, t) *
                                             R(i, q) * R(j, r) * R(k, s) *
                                             R(l, t);
}
/******************************************************************************/
template <typename DataT>
void
FM::changeBasis(
    minitensor::Tensor3<DataT, FM::THREE_D>&       inMatlBasis,
    const minitensor::Tensor3<DataT, FM::THREE_D>& inGlobalBasis,
    const minitensor::Tensor<DataT, FM::THREE_D>&  R)
/******************************************************************************/
{
  int num_dims = R.get_dimension();
  inMatlBasis.clear();
  for (int i = 0; i < num_dims; i++)
    for (int j = 0; j < num_dims; j++)
      for (int k = 0; k < num_dims; k++)
        for (int q = 0; q < num_dims; q++)
          for (int r = 0; r < num_dims; r++)
            for (int s = 0; s < num_dims; s++)
              inMatlBasis(i, j, k) +=
                  inGlobalBasis(q, r, s) * R(i, q) * R(j, r) * R(k, s);
}
/******************************************************************************/
template <typename DataT>
void
FM::changeBasis(
    minitensor::Tensor<DataT, FM::THREE_D>&       inMatlBasis,
    const minitensor::Tensor<DataT, FM::THREE_D>& inGlobalBasis,
    const minitensor::Tensor<DataT, FM::THREE_D>& R)
/******************************************************************************/
{
  int num_dims = R.get_dimension();
  inMatlBasis.clear();
  for (int i = 0; i < num_dims; i++)
    for (int j = 0; j < num_dims; j++)
      for (int q = 0; q < num_dims; q++)
        for (int r = 0; r < num_dims; r++)
          inMatlBasis(i, j) += inGlobalBasis(q, r) * R(i, q) * R(j, r);
}
/******************************************************************************/
template <typename DataT>
void
FM::changeBasis(
    minitensor::Vector<DataT, FM::THREE_D>&       inMatlBasis,
    const minitensor::Vector<DataT, FM::THREE_D>& inGlobalBasis,
    const minitensor::Tensor<DataT, FM::THREE_D>& R)
/******************************************************************************/
{
  int num_dims = R.get_dimension();
  inMatlBasis.clear();
  for (int i = 0; i < num_dims; i++)
    for (int q = 0; q < num_dims; q++)
      inMatlBasis(i) += inGlobalBasis(q) * R(i, q);
}

/******************************************************************************/
template <typename NLS, typename DataT>
void
FM::DescentNorm(NLS& nls, minitensor::Vector<DataT, FM::MAX_TRNS>& xi)
{
}
/******************************************************************************/

/******************************************************************************/
template <typename NLS, typename DataT>
void
FM::ScaledDescent(NLS& nls, minitensor::Vector<DataT, FM::MAX_TRNS>& xi)
/******************************************************************************/
{
  //  minitensor::Vector<ArgT, FM::MAX_TRNS>
  //  residual = nls.gradient(xi);
}

/******************************************************************************/
template <typename DataT, typename ArgT>
void
FM::computeBinFractions(
    minitensor::Vector<ArgT, FM::MAX_TRNS> const& xi,
    Teuchos::Array<ArgT>&                         newFractions,
    Teuchos::Array<DataT> const&                  oldFractions,
    Teuchos::Array<int> const&                    transitionMap,
    Kokkos::DynRankView<DataT> const&             aMatrix)
/******************************************************************************/
{
  int nVariants    = oldFractions.size();
  int nTransitions = transitionMap.size();
  for (int I = 0; I < nVariants; I++) {
    newFractions[I] = oldFractions[I];
    for (int i = 0; i < nTransitions; i++) {
      if (transitionMap[i] >= 0) {
        newFractions[I] += xi(transitionMap[i]) * aMatrix(I, i);
      }
    }
  }
}

/******************************************************************************/
template <typename ArgT>
void
FM::computeInitialState(
    Teuchos::Array<RealType> const&              fractions,
    Teuchos::Array<FM::CrystalVariant> const&    crystalVariants,
    minitensor::Tensor<ArgT, FM::THREE_D> const& x,
    minitensor::Tensor<ArgT, FM::THREE_D>&       X,
    minitensor::Tensor<ArgT, FM::THREE_D>&       linear_x,
    minitensor::Vector<ArgT, FM::THREE_D> const& E,
    minitensor::Vector<ArgT, FM::THREE_D>&       D,
    minitensor::Vector<ArgT, FM::THREE_D>&       linear_D)
/******************************************************************************/
{
  minitensor::Tensor4<ArgT, FM::THREE_D> C;
  C.clear();
  minitensor::Tensor3<ArgT, FM::THREE_D> h;
  h.clear();
  minitensor::Tensor<ArgT, FM::THREE_D> b;
  b.clear();

  minitensor::Tensor<ArgT, FM::THREE_D> remanent_x;
  remanent_x.clear();
  minitensor::Vector<ArgT, FM::THREE_D> remanent_D;
  remanent_D.clear();

  int nVariants = crystalVariants.size();
  for (int i = 0; i < nVariants; i++) {
    const CrystalVariant& variant = crystalVariants[i];
    remanent_x += fractions[i] * variant.spontStrain;
    remanent_D += fractions[i] * variant.spontEDisp;
    C += fractions[i] * variant.C;
    h += fractions[i] * variant.h;
    b += fractions[i] * variant.b;
  }

  minitensor::Tensor<ArgT, FM::THREE_D> eps = minitensor::inverse(b);

  linear_x = x - remanent_x;

  linear_D = dot(eps, E + dotdot(h, linear_x));
  X        = dotdot(C, linear_x) - dot(linear_D, h);

  D = linear_D + remanent_D;
}

/******************************************************************************/
template <typename ArgT>
void
FM::computeRelaxedState(
    Teuchos::Array<ArgT> const&                  fractions,
    Teuchos::Array<FM::CrystalVariant> const&    crystalVariants,
    minitensor::Tensor<ArgT, FM::THREE_D> const& x,
    minitensor::Tensor<ArgT, FM::THREE_D>&       X,
    minitensor::Tensor<ArgT, FM::THREE_D>&       linear_x,
    minitensor::Vector<ArgT, FM::THREE_D>&       E,
    minitensor::Vector<ArgT, FM::THREE_D> const& D,
    minitensor::Vector<ArgT, FM::THREE_D>&       linear_D)
/******************************************************************************/
{
  minitensor::Tensor4<ArgT, FM::THREE_D> C;
  C.clear();
  minitensor::Tensor3<ArgT, FM::THREE_D> h;
  h.clear();
  minitensor::Tensor<ArgT, FM::THREE_D> b;
  b.clear();

  minitensor::Tensor<ArgT, FM::THREE_D> remanent_x;
  remanent_x.clear();
  minitensor::Vector<ArgT, FM::THREE_D> remanent_D;
  remanent_D.clear();

  int nVariants = crystalVariants.size();
  for (int i = 0; i < nVariants; i++) {
    const CrystalVariant& variant = crystalVariants[i];
    remanent_x += fractions[i] * variant.spontStrain;
    remanent_D += fractions[i] * variant.spontEDisp;
    C += fractions[i] * variant.C;
    h += fractions[i] * variant.h;
    b += fractions[i] * variant.b;
  }

  linear_x = x - remanent_x;
  linear_D = D - remanent_D;

  X = dotdot(C, linear_x) - dot(linear_D, h);
  E = dotdot(h, linear_x) + dot(b, linear_D);
}

/******************************************************************************/
template <typename DataT, typename ArgT>
void
FM::computeResidual(
    minitensor::Vector<ArgT, FM::MAX_TRNS>&      residual,
    Teuchos::Array<ArgT> const&                  fractions,
    Teuchos::Array<int> const&                   transitionMap,
    Teuchos::Array<FM::Transition> const&        transitions,
    Teuchos::Array<FM::CrystalVariant> const&    crystalVariants,
    Teuchos::Array<DataT> const&                 tBarrier,
    Kokkos::DynRankView<DataT> const&            aMatrix,
    minitensor::Tensor<ArgT, FM::THREE_D> const& X,
    minitensor::Tensor<ArgT, FM::THREE_D> const& linear_x,
    minitensor::Vector<ArgT, FM::THREE_D> const& E,
    minitensor::Vector<ArgT, FM::THREE_D> const& linear_D)
/******************************************************************************/
{
  int  nVariants = fractions.size();
  ArgT half      = 1.0 / 2.0;
  for (int I = 0; I < nVariants; I++) {
    ArgT fracI = fractions[I];
    for (int J = 0; J < nVariants; J++) {
      int i = I * nVariants + J;
      if (transitionMap[i] >= 0) {
        const Transition& transition = transitions[i];
        int               lindex     = transitionMap[i];
        residual[lindex] = -tBarrier[i] - dotdot(transition.transStrain, X) -
                           dot(transition.transEDisp, E);
      }
    }
  }
  int nTransitions = transitions.size();
  for (int I = 0; I < nVariants; I++) {
    ArgT                  myRate(0.0);
    const CrystalVariant& variant = crystalVariants[I];
    myRate +=
        dotdot(
            linear_x, dotdot(variant.C, linear_x) - dot(linear_D, variant.h)) *
        half;
    myRate +=
        dot(linear_D, dot(variant.b, linear_D) - dotdot(variant.h, linear_x)) *
        half;
    for (int i = 0; i < nTransitions; i++) {
      if (transitionMap[i] >= 0) {
        DataT aVal = aMatrix(I, i);
        if (aVal == 0.0) continue;
        int lindex = transitionMap[i];
        residual[lindex] += aVal * myRate;
      }
    }
  }
}
