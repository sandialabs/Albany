#ifndef ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP
#define ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP

#include "Albany_ThyraTypes.hpp"

namespace Albany
{

// An Albany-owned combine mode enum, so we don't stick to either  or Epetra
// (note:: it would be nice if Teuchos provided such enumeration)
enum class CombineMode {
  ADD,    // Add remote contributions to local ones
  INSERT, // Replace local contributions with remote ones (beware of race conditions!)
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
  Teuchos::Array<int> getGhostedAuraOwners () const;

  // Get a LID-rank par for all ids in the owned aura VS. A pair <a,b> means that rank b will need lid a.
  Teuchos::Array<std::pair<LO,int>> getOwnedAuraUsers () const;

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
  mutable Teuchos::Array<int> ghosted_aura_owners;
  mutable Teuchos::Array<std::pair<LO,int>> owned_aura_users;
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

inline Teuchos::Array<int>
CombineAndScatterManager::getGhostedAuraOwners () const {
  if (ghosted_aura_owners.size()==0) {
    create_ghosted_aura_owners();
  }
  return ghosted_aura_owners;
}

inline Teuchos::Array<std::pair<LO,int>>
CombineAndScatterManager::getOwnedAuraUsers () const {
  if (owned_aura_users.size()==0) {
    create_owned_aura_users();
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
