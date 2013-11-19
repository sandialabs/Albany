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
  virtual void PrintSelf(ostream &os, vtkIndent indent);

  void SetArrayRCP(const Teuchos::ArrayRCP<Scalar> &array, int numComps);

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

  // Description:
  // This container is read only -- this method does nothing but print a
  // warning.
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
TeuchosArrayRCPDataArray<Scalar>::GetValue(vtkIdType idx)
{
  return this->Data[idx];
}

template <typename Scalar> inline Scalar &
TeuchosArrayRCPDataArray<Scalar>::GetValueReference(vtkIdType idx)
{
  return this->Data[idx];
}

template <typename Scalar> inline void
TeuchosArrayRCPDataArray<Scalar>::GetTupleValue(
    vtkIdType idx, TeuchosArrayRCPDataArray::ValueType *t)
{
  int comps = this->NumberOfComponents;
  typename ContainerType::const_iterator tuple
      = this->Data.begin() + (idx * comps);
  while (comps-- > 0)
    *(t++) = *(tuple++);
}

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYST_TEUCHOSARRAYRCPDATAARRAY
