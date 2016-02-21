/*----------------------------------------------------------------------------
 * Copyright © Microsoft Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not 
 * use this file except in compliance with the License.  You may obtain a copy 
 * of the License at http://www.apache.org/licenses/LICENSE-2.0  
 * 
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED 
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, 
 * MERCHANTABLITY OR NON-INFRINGEMENT. 
 *
 * See the Apache Version 2.0 License for specific language governing 
 * permissions and limitations under the License.
 *---------------------------------------------------------------------------
 * 
 * amax_test_list.h
 *
 * Testing framework component used to register tests.
 *
 *---------------------------------------------------------------------------*/

#pragma once

#include <vector>
#include <memory>

//
// test list
//

class test_list_item
{
public:
    ~test_list_item() {}
    virtual void run_all_tests() = 0;
};

template <typename test_type>
class test_register_helper
{
public:
	test_register_helper()
	{
		get_test_list().push_back( std::make_shared<test_type>() );
	}
};

// helper macro to create an "anonymous" variable to add to the global test list
#define REGISTER_TEST(test_name,value_type) test_register_helper<test_name<value_type>> TEST_##test_name##value_type;

typedef std::vector<std::shared_ptr<test_list_item>> test_list;

test_list& get_test_list();
void execute_all_tests();
