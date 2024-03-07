#ifndef ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP
#define ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP

#include "Albany_ThyraTypes.hpp"

namespace Albany
{

// An Albany-owned combine mode enum
enum class CombineMode {
  ADD,    // Add remote contributions to local ones
  INSERT, // Replace local contributions with remote ones (beware of race conditions!)
  ABSMAX, // Replace local contributions with max(|local|,|remote|)
  ZERO    // Ignore remote contributions
};

// This class is intended to hide the implementation details regarding
// how the underlying linear algebra library deals with combining/scattering
// distributed objects. The interface accepts Thyra objects (vectors,
// multivectors, linear operators), and in the hidden implementation,
// these are casted to the concrete linear algebra structures, and
// the corresponding combine/scatter method is used
class CombineAndScatterManager
{
public:

  CombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                            const Teuchos::RCP<const Thyra_VectorSpace>& overlapped);

  virtual ~CombineAndScatterManager () = default;

  // The VS corresponding to elements exclusively owned by this rank
  Teuchos::RCP<const Thyra_VectorSpace> getOwnedVectorSpace () const { return owned_vs; }

  // The VS corresponding to elements that this rank owns or shares with other ranks
  Teuchos::RCP<const Thyra_VectorSpace> getOverlappedVectorSpace () const { return overlapped_vs;  }

  // The subset of the overlapped VS that is shared by at least another rank, that is
  //    sharedVS(rank=p) = overlappedVS(rank=p) intersect Sum_q(overlappedVS(rank=q))
  Teuchos::RCP<const Thyra_VectorSpace> getSharedAuraVectorSpace () const;

  // The subset of the shared aura VS that also belongs to owned VS, that is
  //   ownedAuraVS = sharedAuraVS intersect ownedVS
  Teuchos::RCP<const Thyra_VectorSpace> getOwnedAuraVectorSpace () const;

  // The subset of the shared aura VS that is not in owned aura VS
  //    ghostedVS = sharedAuraVS minus ownedAuraVS
  Teuchos::RCP<const Thyra_VectorSpace> getGhostedAuraVectorSpace () const;

  // Get the ranks that own ids in the ghosted aura VS
  const Teuchos::Array<int>& getGhostedAuraOwners () const;

  // Get an array of GID-rank pairs. A pair <gid,pid> means that rank pid will need global element gid.
  const Teuchos::Array<std::pair<GO,int>>& getOwnedAuraUsers () const;

  // Combine methods
  virtual void combine (const Thyra_Vector& src,
                              Thyra_Vector& dst,
                        const CombineMode CM) const = 0;
  virtual void combine (const Thyra_MultiVector& src,
                              Thyra_MultiVector& dst,
                        const CombineMode CM) const = 0;
  virtual void combine (const Thyra_LinearOp& src,
                              Thyra_LinearOp& dst,
                        const CombineMode CM) const = 0;

  virtual void combine (const Teuchos::RCP<const Thyra_Vector>& src,
                        const Teuchos::RCP<      Thyra_Vector>& dst,
                        const CombineMode CM) const = 0;
  virtual void combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
                        const Teuchos::RCP<      Thyra_MultiVector>& dst,
                        const CombineMode CM) const = 0;
  virtual void combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
                        const Teuchos::RCP<      Thyra_LinearOp>& dst,
                        const CombineMode CM) const = 0;

  // Scatter methods
  virtual void scatter (const Thyra_Vector& src,
                              Thyra_Vector& dst,
                        const CombineMode CM) const = 0;
  virtual void scatter (const Thyra_MultiVector& src,
                              Thyra_MultiVector& dst,
                        const CombineMode CM) const = 0;
  virtual void scatter (const Thyra_LinearOp& src,
                              Thyra_LinearOp& dst,
                        const CombineMode CM) const = 0;

  virtual void scatter (const Teuchos::RCP<const Thyra_Vector>& src,
                        const Teuchos::RCP<      Thyra_Vector>& dst,
                        const CombineMode CM) const = 0;
  virtual void scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
                        const Teuchos::RCP<      Thyra_MultiVector>& dst,
                        const CombineMode CM) const = 0;
  virtual void scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
                        const Teuchos::RCP<      Thyra_LinearOp>& dst,
                        const CombineMode CM) const = 0;

protected:

  void create_aura_vss () const;
  virtual void create_ghosted_aura_owners () const = 0;
  virtual void create_owned_aura_users () const = 0;

  Teuchos::RCP<const Thyra_VectorSpace>   owned_vs;
  Teuchos::RCP<const Thyra_VectorSpace>   overlapped_vs;

  // Mutable, so we can lazy initialize them
  mutable Teuchos::RCP<const Thyra_VectorSpace>   owned_aura_vs;
  mutable Teuchos::RCP<const Thyra_VectorSpace>   shared_aura_vs;
  mutable Teuchos::RCP<const Thyra_VectorSpace>   ghosted_aura_vs;
  mutable Teuchos::Array<int>                     ghosted_aura_owners;
  mutable Teuchos::Array<std::pair<GO,int>>       owned_aura_users;

  // Note: we do need these flags, since simply checking the size of the arrays
  //       would not be enough. In fact, if one rank has the array of size 0,
  //       the method to compute the array would be called multiple times,
  //       which is wrong if the method involves global communication.
  //       The alternative would be to store RCP to the array, but it
  //       makes the code a bit ugly.
  mutable bool ghosted_aura_owners_computed;
  mutable bool owned_aura_users_computed;
};

inline Teuchos::RCP<const Thyra_VectorSpace>
CombineAndScatterManager::getSharedAuraVectorSpace () const {
  if (shared_aura_vs.is_null()) {
    create_aura_vss();
  }
  return shared_aura_vs;
}

inline Teuchos::RCP<const Thyra_VectorSpace>
CombineAndScatterManager::getOwnedAuraVectorSpace  () const {
  if (owned_aura_vs.is_null()) {
    create_aura_vss();
  }
  return owned_aura_vs;
}

inline Teuchos::RCP<const Thyra_VectorSpace>
CombineAndScatterManager::getGhostedAuraVectorSpace  () const {
  if (ghosted_aura_vs.is_null()) {
    create_aura_vss();
  }
  return ghosted_aura_vs;
}

inline const Teuchos::Array<int>&
CombineAndScatterManager::getGhostedAuraOwners () const {
  if (!ghosted_aura_owners_computed) {
    create_ghosted_aura_owners();
    ghosted_aura_owners_computed = true;
  }
  return ghosted_aura_owners;
}

inline const Teuchos::Array<std::pair<GO,int>>&
CombineAndScatterManager::getOwnedAuraUsers () const {
  if (!owned_aura_users_computed) {
    create_owned_aura_users();
    owned_aura_users_computed = true;
  }
  return owned_aura_users;
}

// Utility function that returns a concrete manager, depending on the concrete type
// of the input vector spaces.
Teuchos::RCP<CombineAndScatterManager>
createCombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                const Teuchos::RCP<const Thyra_VectorSpace>& overlapped);

} // namespace Albany

#endif // ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP
