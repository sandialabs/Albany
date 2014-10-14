//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(AAdapt_RandomCriterion_hpp)
#define AAdapt_RandomCriterion_hpp

#include "Fracture.h"
#include "Albany_STKDiscretization.hpp"

namespace AAdapt {

///
/// \brief Random fracture criterion
///
/// This class contains the abstract interface for determining if
/// fracture has occurred between two elements, based on the average
/// stress in the elements sharing the edge between them.
///
///
class RandomCriterion: public LCM::AbstractFractureCriterion {

  public:

    ///
    /// \brief Default constructor for the criterion object
    ///
    RandomCriterion(int num_dim,
                    Albany::STKDiscretization& stk);


    ///
    /// \brief Stress fracture criterion function.
    ///
    /// \param[in] entity
    /// \param[in] probability
    /// \return is criterion met
    ///
    /// Given an entity and probability, will determine if fracture
    /// criterion is met. Will return true if fracture criterion is
    /// met, else false.  Fracture only defined on surface of
    /// elements. Thus, input entity must be of rank dimension-1, else
    /// error. For 2D, entity rank must = 1.  For 3D, entity rank must
    /// = 2.
    ///
    bool
    computeFractureCriterion(stk::mesh::Entity entity, double p);

  private:

    RandomCriterion();
    RandomCriterion(const RandomCriterion&);
    RandomCriterion& operator=(const RandomCriterion&);

    Albany::STKDiscretization& stk_;

}; // class RandomCriterion


} // namespace AAdapt

#endif // RandomCriterion_hpp
