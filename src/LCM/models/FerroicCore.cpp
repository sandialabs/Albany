#include "FerroicCore.hpp"

/******************************************************************************/
FM::CrystalPhase::CrystalPhase(
    minitensor::Tensor<RealType, FM::THREE_D>&  matBasis,
    minitensor::Tensor4<RealType, FM::THREE_D>& C_matBasis,
    minitensor::Tensor3<RealType, FM::THREE_D>& h_matBasis,
    minitensor::Tensor<RealType, FM::THREE_D>&  e_matBasis)
/******************************************************************************/
{
  FM::changeBasis(C, C_matBasis, matBasis);
  FM::changeBasis(h, h_matBasis, matBasis);

  minitensor::Tensor<RealType, FM::THREE_D> b_matBasis;
  b_matBasis = minitensor::inverse(e_matBasis);
  FM::changeBasis(b, b_matBasis, matBasis);

  basis = matBasis;
}
