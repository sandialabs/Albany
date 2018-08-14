//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(FM_AlbanyInterface_hpp)
#define FM_AlbanyInterface_hpp

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado.hpp"

#include "FerroicModel.hpp"

namespace LCM {

//! \brief Ferroic model for electromechanics
template <typename EvalT, typename Traits>
class FerroicDriver : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  using LCM::ConstitutiveModel<EvalT, Traits>::num_dims_;
  using LCM::ConstitutiveModel<EvalT, Traits>::num_pts_;
  using LCM::ConstitutiveModel<EvalT, Traits>::field_name_map_;
  using LCM::ConstitutiveModel<EvalT, Traits>::weights_;

  ///
  /// Constructor
  ///
  FerroicDriver(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Destructor
  ///
  virtual ~FerroicDriver(){};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  // Kokkos
  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

 private:
  Teuchos::RCP<FM::FerroicModel<EvalT>> ferroicModel;

  ///
  /// Private to prohibit copying
  ///
  FerroicDriver(const FerroicDriver&);

  ///
  /// Private to prohibit copying
  ///
  FerroicDriver&
  operator=(const FerroicDriver&);

  ///
  /// INDEPENDENT FIELD NAMES
  ///
  std::string strainName;
  std::string efieldName;

  ///
  /// EVALUATED FIELD NAMES
  ///
  std::string stressName;
  std::string edispName;

  ///
  /// STATE VARIABLE NAMES
  ///
  Teuchos::Array<std::string> binNames;
};

void
parseBasis(
    const Teuchos::ParameterList&              pBasis,
    minitensor::Tensor<RealType, FM::THREE_D>& R);
void
parseTensor4(
    const Teuchos::ParameterList&               pConsts,
    minitensor::Tensor4<RealType, FM::THREE_D>& tensor);
void
parseTensor3(
    const Teuchos::ParameterList&               pConsts,
    minitensor::Tensor3<RealType, FM::THREE_D>& tensor);
void
parseTensor(
    const Teuchos::ParameterList&              pConsts,
    minitensor::Tensor<RealType, FM::THREE_D>& tensor);

FM::CrystalVariant
parseCrystalVariant(
    const Teuchos::Array<Teuchos::RCP<FM::CrystalPhase>>& phases,
    const Teuchos::ParameterList&                         vParam);

}  // namespace LCM

#endif
