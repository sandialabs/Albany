//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <gtest/gtest.h>
#include <iostream>
#include "../../../utility/StaticAllocator.hpp"

using namespace utility;

namespace {
template <std::size_t N>
struct TestArray
{
  unsigned char d[N];
};

TEST(StaticAllocatorTest, SimpleAllocation)
{
  StaticAllocator alloc(1024);

  auto tarray = alloc.create<TestArray<1024>>();

  ASSERT_NE(tarray, StaticPointer<TestArray<1024>>());
}

TEST(StaticAllocatorTest, FailAllocation)
{
  StaticAllocator alloc(1024);

  ASSERT_THROW(alloc.create<TestArray<1025>>(), std::bad_alloc);
}

TEST(StaticAllocatorTest, MultipleAllocation)
{
  StaticAllocator alloc(1024);

  auto tarray1 = alloc.create<TestArray<256>>();
  ASSERT_NE(tarray1, StaticPointer<TestArray<256>>());

  auto tarray2 = alloc.create<TestArray<512>>();
  ASSERT_NE(tarray2, StaticPointer<TestArray<512>>());

  auto tarray3 = alloc.create<TestArray<256>>();
  ASSERT_NE(tarray3, StaticPointer<TestArray<256>>());
}

TEST(StaticAllocatorTest, MultipleFail)
{
  StaticAllocator alloc(1024);

  auto tarray1 = alloc.create<TestArray<257>>();
  ASSERT_NE(tarray1, StaticPointer<TestArray<257>>());

  auto tarray2 = alloc.create<TestArray<512>>();
  ASSERT_NE(tarray2, StaticPointer<TestArray<512>>());

  ASSERT_THROW(alloc.create<TestArray<256>>(), std::bad_alloc);

  auto tarray3 = alloc.create<TestArray<255>>();
  ASSERT_NE(tarray3, StaticPointer<TestArray<255>>());
}

struct PointerTester
{
  PointerTester(bool* ptr) : active(ptr) { *active = true; }
  ~PointerTester() { *active = false; }

  bool* active;
};

using PTest = StaticPointer<PointerTester>;

TEST(StaticPointerTest, Destroy)
{
  StaticAllocator alloc(128);
  bool            active = false;

  {
    auto p = alloc.create<PointerTester>(&active);

    ASSERT_TRUE(active);
  }

  ASSERT_FALSE(active);
}

TEST(StaticPointerTest, Move)
{
  StaticAllocator alloc(128);
  bool            active = false;

  {
    auto p = alloc.create<PointerTester>(&active);

    ASSERT_TRUE(active);

    // Move operation
    auto q = std::move(p);

    ASSERT_TRUE(active);
    ASSERT_NE(q, PTest());
    ASSERT_EQ(p, PTest());
  }

  ASSERT_FALSE(active);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
