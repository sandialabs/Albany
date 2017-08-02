//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CATALYST_EPETRADATAARRAY
#define ALBANY_CATALYST_EPETRADATAARRAY

#include "vtkMappedDataArray.h"

#include <Epetra_Vector.h>

namespace Albany {
namespace Catalyst {

class EpetraDataArray: public vtkMappedDataArray<double>
{
public:
  vtkAbstractTypeMacro(EpetraDataArray, vtkMappedDataArray<double>)
  vtkMappedDataArrayNewInstanceMacro(EpetraDataArray)
  static EpetraDataArray *New();
  virtual void PrintSelf(ostream &os, vtkIndent indent) override;

  void SetEpetraVector(const Epetra_Vector &vector);

  // Reimplemented virtuals -- see superclasses for descriptions:
  void Initialize() override;
  void GetTuples(vtkIdList *ptIds, vtkAbstractArray *output) override;
  void GetTuples(vtkIdType p1, vtkIdType p2, vtkAbstractArray *output) override;
  void Squeeze() override;
  vtkArrayIterator *NewIterator() override;
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
  void GetTypedTuple(vtkIdType idx, ValueType *t) const override;

  // This container is read only. These methods do nothing but print a warning.
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
  void SetTypedTuple(vtkIdType i, const ValueType *t) override;
  void InsertTypedTuple(vtkIdType i, const ValueType *t) override;
  vtkIdType InsertNextTypedTuple(const ValueType *t) override;
  void SetValue(vtkIdType idx, ValueType value) override;
  vtkIdType InsertNextValue(ValueType v) override;
  void InsertValue(vtkIdType idx, ValueType v) override;

protected:
  EpetraDataArray();
  ~EpetraDataArray();

  const Epetra_Vector *Data;

private:
  EpetraDataArray(const EpetraDataArray&); // Not implemented.
  void operator=(const EpetraDataArray&);  // Not implemented.

  vtkIdType Lookup(double val, vtkIdType startIdx);

  double TmpDouble;
};

inline EpetraDataArray::ValueType EpetraDataArray::GetValue(vtkIdType idx) const
{
  return (*this->Data)[idx];
}

inline EpetraDataArray::ValueType &
EpetraDataArray::GetValueReference(vtkIdType idx)
{
  // Bad bad bad. VTK has no concept of 'const', so we'll just cross our fingers
  // that no one writes to the returned reference.
  return *const_cast<double*>(&(*this->Data)[idx]);
}

inline void
EpetraDataArray::GetTypedTuple(vtkIdType idx, EpetraDataArray::ValueType *t) const
{
  *t = (*this->Data)[idx];
}

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYST_EPETRADATAARRAY
