//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StringUtils.hpp"

#include <Teuchos_UnitTestHarness.hpp>

namespace Albany
{

TEUCHOS_UNIT_TEST(StringUtils,strint)
{
  std::string s = "foo";
  TEST_ASSERT (util::strint(s,0)=="foo 0");
}

TEUCHOS_UNIT_TEST(StringUtils,join)
{
  std::vector<int> v = {-1,0,1};
  auto s = util::join(v,",");
  TEST_ASSERT (s=="-1,0,1");

  TEST_ASSERT (util::join({1},"BLAHBLAH")=="1");
}

TEUCHOS_UNIT_TEST(StringUtils,ParseList)
{
  std::string valid_1 = "[a]";
  std::string valid_2 = "[a,b]";
  std::string valid_3 = "[a,[b,c]]";
  std::string valid_4 = "[[a,b],c]";
  std::string valid_5 = "[[a]]";
  std::string valid_6 = "[[a,[b,c]],d]";

  std::string invalid_1 = "[]";
  std::string invalid_2 = "[,b]";
  std::string invalid_3 = "[a[b,c]]";
  std::string invalid_4 = "[,,c]";
  std::string invalid_5 = "[a,]";

  using namespace util;

  // Ensure we can tell valid from invalid strings
  TEST_ASSERT (validNestedListFormat(valid_1));
  TEST_ASSERT (validNestedListFormat(valid_2));
  TEST_ASSERT (validNestedListFormat(valid_3));
  TEST_ASSERT (validNestedListFormat(valid_4));
  TEST_ASSERT (validNestedListFormat(valid_5));
  TEST_ASSERT (validNestedListFormat(valid_6));
  
  TEST_ASSERT (not validNestedListFormat(invalid_1));
  TEST_ASSERT (not validNestedListFormat(invalid_2));
  TEST_ASSERT (not validNestedListFormat(invalid_3));
  TEST_ASSERT (not validNestedListFormat(invalid_4));
  TEST_ASSERT (not validNestedListFormat(invalid_5));

  // Parse strings into lists
  auto pl_1 = parseNestedList(valid_1);
  auto pl_2 = parseNestedList(valid_2);
  auto pl_3 = parseNestedList(valid_3);
  auto pl_4 = parseNestedList(valid_4);
  auto pl_5 = parseNestedList(valid_5);
  auto pl_6 = parseNestedList(valid_6);

  // Test counters
  TEST_ASSERT (pl_1.get<int>("Num Entries")==1);
  TEST_ASSERT (pl_2.get<int>("Num Entries")==2);
  TEST_ASSERT (pl_3.get<int>("Num Entries")==2);
  TEST_ASSERT (pl_4.get<int>("Num Entries")==2);
  TEST_ASSERT (pl_5.get<int>("Num Entries")==1);
  TEST_ASSERT (pl_6.get<int>("Num Entries")==2);

  TEST_ASSERT (pl_1.get<int>("Depth")==1);
  TEST_ASSERT (pl_2.get<int>("Depth")==1);
  TEST_ASSERT (pl_3.get<int>("Depth")==2);
  TEST_ASSERT (pl_4.get<int>("Depth")==2);
  TEST_ASSERT (pl_5.get<int>("Depth")==2);
  TEST_ASSERT (pl_6.get<int>("Depth")==3);

  // For entries, only test valid_6 = "[[a,[b,c]],d]";
  TEST_ASSERT (pl_6.get<std::string>("Type 0")=="List");
  TEST_ASSERT (pl_6.get<std::string>("Type 1")=="Value");
  TEST_ASSERT (pl_6.get<std::string>("Entry 1")=="d");

  const auto& sub = pl_6.sublist("Entry 0");

  TEST_ASSERT (sub.get<int>("Num Entries")==2);
  TEST_ASSERT (sub.get<int>("Depth")==2);
  TEST_ASSERT (sub.get<std::string>("Type 0")=="Value");
  TEST_ASSERT (sub.get<std::string>("Type 1")=="List");
  TEST_ASSERT (sub.get<std::string>("Entry 0")=="a");

  const auto& subsub = sub.sublist("Entry 1");
  TEST_ASSERT (subsub.get<int>("Num Entries")==2);
  TEST_ASSERT (subsub.get<int>("Depth")==1);
  TEST_ASSERT (subsub.get<std::string>("Type 0")=="Value");
  TEST_ASSERT (subsub.get<std::string>("Type 1")=="Value");
  TEST_ASSERT (subsub.get<std::string>("Entry 0")=="b");
  TEST_ASSERT (subsub.get<std::string>("Entry 1")=="c");
}

} // namespace Albany
