#include "FerroicCore.hpp"
 
/******************************************************************************/
FM::CrystalPhase::
CrystalPhase(Intrepid2::Tensor <RealType, FM::THREE_D>& matBasis, 
             Intrepid2::Tensor4<RealType, FM::THREE_D>& C_matBasis, 
             Intrepid2::Tensor3<RealType, FM::THREE_D>& h_matBasis, 
             Intrepid2::Tensor <RealType, FM::THREE_D>& e_matBasis)
/******************************************************************************/
{
  FM::changeBasis(C, C_matBasis, matBasis);
  FM::changeBasis(h, h_matBasis, matBasis);

  Intrepid2::Tensor<RealType, FM::THREE_D> b_matBasis;
  b_matBasis = Intrepid2::inverse(e_matBasis);
  FM::changeBasis(b, b_matBasis, matBasis);

  basis = matBasis;
}
