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
  virtual void PrintSelf(ostream &os, vtkIndent indent);

  void SetEpetraVector(const Epetra_Vector &vector);

  // Reimplemented virtuals -- see superclasses for descriptions:
  void Initialize();
  void GetTuples(vtkIdList *ptIds, vtkAbstractArray *output);
  void GetTuples(vtkIdType p1, vtkIdType p2, vtkAbstractArray *output);
  void Squeeze();
  vtkArrayIterator *NewIterator();
  vtkIdType LookupValue(vtkVariant value);
  void LookupValue(vtkVariant value, vtkIdList *ids);
  vtkVariant GetVariantValue(vtkIdType idx);
  void ClearLookup();
  double* GetTuple(vtkIdType i);
  void GetTuple(vtkIdType i, double *tuple);
  vtkIdType LookupTypedValue(ValueType value);
  void LookupTypedValue(ValueType value, vtkIdList *ids);
  ValueType GetValue(vtkIdType idx);
  ValueType& GetValueReference(vtkIdType idx);
  void GetTupleValue(vtkIdType idx, ValueType *t);

  // This container is read only. These methods do nothing but print a warning.
  int Allocate(vtkIdType sz, vtkIdType ext);
  int Resize(vtkIdType numTuples);
  void SetNumberOfTuples(vtkIdType number);
  void SetTuple(vtkIdType i, vtkIdType j, vtkAbstractArray *source);
  void SetTuple(vtkIdType i, const float *source);
  void SetTuple(vtkIdType i, const double *source);
  void InsertTuple(vtkIdType i, vtkIdType j, vtkAbstractArray *source);
  void InsertTuple(vtkIdType i, const float *source);
  void InsertTuple(vtkIdType i, const double *source);
  void InsertTuples(vtkIdList *dstIds, vtkIdList *srcIds,
                    vtkAbstractArray *source);
  vtkIdType InsertNextTuple(vtkIdType j, vtkAbstractArray *source);
  vtkIdType InsertNextTuple(const float *source);
  vtkIdType InsertNextTuple(const double *source);
  void DeepCopy(vtkAbstractArray *aa);
  void DeepCopy(vtkDataArray *da);
  void InterpolateTuple(vtkIdType i, vtkIdList *ptIndices,
                        vtkAbstractArray* source,  double* weights);
  void InterpolateTuple(vtkIdType i, vtkIdType id1, vtkAbstractArray *source1,
                        vtkIdType id2, vtkAbstractArray *source2, double t);
  void SetVariantValue(vtkIdType idx, vtkVariant value);
  void RemoveTuple(vtkIdType id);
  void RemoveFirstTuple();
  void RemoveLastTuple();
  void SetTupleValue(vtkIdType i, const ValueType *t);
  void InsertTupleValue(vtkIdType i, const ValueType *t);
  vtkIdType InsertNextTupleValue(const ValueType *t);
  void SetValue(vtkIdType idx, ValueType value);
  vtkIdType InsertNextValue(ValueType v);
  void InsertValue(vtkIdType idx, ValueType v);

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

inline EpetraDataArray::ValueType EpetraDataArray::GetValue(vtkIdType idx)
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
EpetraDataArray::GetTupleValue(vtkIdType idx, EpetraDataArray::ValueType *t)
{
  *t = (*this->Data)[idx];
}

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYST_EPETRADATAARRAY
