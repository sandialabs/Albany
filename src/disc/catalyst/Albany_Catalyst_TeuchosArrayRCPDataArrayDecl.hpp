//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CATALYST_TEUCHOSARRAYRCPDATAARRAY
#define ALBANY_CATALYST_TEUCHOSARRAYRCPDATAARRAY

#include "vtkObject.h"

#include "vtkTypeTemplate.h"    // For templated vtkObject API
#include "vtkMappedDataArray.h" // For mapped data array API/base.
#include "Teuchos_ArrayRCP.hpp" // For wrapped container type.

namespace Albany {
namespace Catalyst {

template <class Scalar>
class TeuchosArrayRCPDataArray:
    public vtkTypeTemplate<TeuchosArrayRCPDataArray<Scalar>,
                           vtkMappedDataArray<Scalar> >
{
public:
  typedef Teuchos::ArrayRCP<Scalar> ContainerType;
  typedef Scalar ValueType;

  vtkMappedDataArrayNewInstanceMacro(TeuchosArrayRCPDataArray<Scalar>)
  static TeuchosArrayRCPDataArray *New();
  virtual void PrintSelf(ostream &os, vtkIndent indent) override;

  void SetArrayRCP(const Teuchos::ArrayRCP<Scalar> &array, int numComps);

  // Reimplemented virtuals -- see superclasses for descriptions:
  void Initialize() override;
  void GetTuples(vtkIdList *ptIds, vtkAbstractArray *output) override;
  void GetTuples(vtkIdType p1, vtkIdType p2, vtkAbstractArray *output) override;
  void Squeeze() override;
  vtkArrayIterator* NewIterator() override;
  vtkIdType LookupValue(vtkVariant value) override;
  void LookupValue(vtkVariant value, vtkIdList *ids) override;
  vtkVariant GetVariantValue(vtkIdType idx) override;
  void ClearLookup() override;
  double* GetTuple(vtkIdType i) override;
  void GetTuple(vtkIdType i, double *tuple) override;
  vtkIdType LookupTypedValue(ValueType value) override;
  void LookupTypedValue(ValueType value, vtkIdList *ids) override;
  ValueType GetValue(vtkIdType idx) const override;
  ValueType& GetValueReference(vtkIdType idx) override;

  // Description:
  // This container is read only -- this method does nothing but print a
  // warning.
  int Allocate(vtkIdType sz, vtkIdType ext) override;
  int Resize(vtkIdType numTuples) override;
  void SetNumberOfTuples(vtkIdType number) override;
  void SetTuple(vtkIdType i, vtkIdType j, vtkAbstractArray *source) override;
  void SetTuple(vtkIdType i, const float *source) override;
  void SetTuple(vtkIdType i, const double *source) override;
  void InsertTuple(vtkIdType i, vtkIdType j, vtkAbstractArray *source) override;
  void InsertTuple(vtkIdType i, const float *source) override;
  void InsertTuple(vtkIdType i, const double *source) override;
  void InsertTuples(vtkIdList *dstIds, vtkIdList *srcIds,
                    vtkAbstractArray *source) override;
  vtkIdType InsertNextTuple(vtkIdType j, vtkAbstractArray *source) override;
  vtkIdType InsertNextTuple(const float *source) override;
  vtkIdType InsertNextTuple(const double *source) override;
  void DeepCopy(vtkAbstractArray *aa) override;
  void DeepCopy(vtkDataArray *da) override;
  void InterpolateTuple(vtkIdType i, vtkIdList *ptIndices,
                        vtkAbstractArray* source,  double* weights) override;
  void InterpolateTuple(vtkIdType i, vtkIdType id1, vtkAbstractArray *source1,
                        vtkIdType id2, vtkAbstractArray *source2, double t) override;
  void SetVariantValue(vtkIdType idx, vtkVariant value) override;
  void RemoveTuple(vtkIdType id) override;
  void RemoveFirstTuple() override;
  void RemoveLastTuple() override;
  void SetValue(vtkIdType idx, ValueType value) override;
  vtkIdType InsertNextValue(ValueType v) override;
  void InsertValue(vtkIdType idx, ValueType v) override;

  void SetTypedTuple(vtkIdType i, const ValueType *t) override;
  void InsertTypedTuple(vtkIdType i, const ValueType *t) override;
  vtkIdType InsertNextTypedTuple(const ValueType *t) override;
  void GetTypedTuple(vtkIdType idx, ValueType *t) const override;
protected:
  TeuchosArrayRCPDataArray();
  ~TeuchosArrayRCPDataArray();

  Teuchos::ArrayRCP<Scalar> Data;

private:
  TeuchosArrayRCPDataArray(const TeuchosArrayRCPDataArray&); // Not implemented.
  void operator=(const TeuchosArrayRCPDataArray &); // Not implemented.

  vtkIdType Lookup(double val, vtkIdType startIdx);

  Scalar *TmpArray; // length = number of components
};

template <typename Scalar> inline Scalar
TeuchosArrayRCPDataArray<Scalar>::GetValue(vtkIdType idx) const
{
  return this->Data[idx];
}

template <typename Scalar> inline Scalar &
TeuchosArrayRCPDataArray<Scalar>::GetValueReference(vtkIdType idx)
{
  return this->Data[idx];
}

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYST_TEUCHOSARRAYRCPDATAARRAY
