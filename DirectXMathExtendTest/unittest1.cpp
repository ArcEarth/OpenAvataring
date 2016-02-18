#include "stdafx.h"
#include "CppUnitTest.h"
#include <DirectXMathExtend.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace DirectX;

namespace DirectXMathExtendTest
{		
	TEST_CLASS(DualQuaternionTest)
	{
	public:
		TEST_METHOD(TranslationTest)
		{
			XMDUALVECTOR dq;
			XMVECTOR epsilon = XMVectorReplicate(0.001f);

			XMVECTOR v = XMVectorSet(1, 0, 0, 0);
			dq = XMDualQuaternionTranslation(XMVectorSet(0, 2, 0, 0));
			v = XMVector3Displacement(v, dq);
			XMVECTOR expected = XMVectorSet(1, 2, 0, 0);
			
			Assert::IsTrue(XMVector3NearEqual(v, expected, epsilon),L"Translation test failed");
			// TODO: Your test code here
		}

		TEST_METHOD(RotationTest)
		{
			XMDUALVECTOR dq;
			XMVECTOR epsilon = XMVectorReplicate(0.001f);

			XMVECTOR v = XMVectorSet(1, 0, 0, 0);
			XMVECTOR q = XMQuaternionRotationRollPitchYaw(1.0, 2, 3);
			dq = XMDualQuaternionRotation(q);

			XMVECTOR expected = XMVector3Rotate(v, q);
			v = XMVector3Displacement(v, dq);

			Assert::IsTrue(XMVector3NearEqual(v, expected, epsilon), L"Rotation test failed");
			// TODO: Your test code here
		}

		TEST_METHOD(RigidTransformTest)
		{
			XMDUALVECTOR dq;
			XMVECTOR epsilon = XMVectorReplicate(0.001f);

			XMVECTOR v = XMVectorSet(1, 0, 0, 0);
			XMVECTOR q = XMQuaternionRotationRollPitchYaw(1.0, 2, 3);
			XMVECTOR t = XMVectorSet(0, 2, 0, 0);
			dq = XMDualQuaternionRotationTranslation(q,t);

			XMVECTOR expected = XMVector3Rotate(v, q);
			expected += t;

			v = XMVector3Displacement(v, dq);

			Assert::IsTrue(XMVector3NearEqual(v, expected, epsilon), L"Rotation test failed");
		}

		TEST_METHOD(RigidTransformConcationTest)
		{
			XMDUALVECTOR dq,dq1;
			XMVECTOR epsilon = XMVectorReplicate(0.001f);

			XMVECTOR v = XMVectorSet(1, 0, 0, 0);

			XMVECTOR q = XMQuaternionRotationRollPitchYaw(1.0, 2, 3);
			XMVECTOR t = XMVectorSet(0, 2, 0, 0);

			XMVECTOR q1 = XMQuaternionRotationRollPitchYaw(3.0, 2, 1);
			XMVECTOR t1 = XMVectorSet(0, -2, -1, 0);

			dq = XMDualQuaternionRotationTranslation(q, t);
			dq1 = XMDualQuaternionRotationTranslation(q1, t1);
			dq = XMDualQuaternionMultipy(dq, dq1);

			XMVECTOR expected = XMVector3Rotate(v, q);
			expected += t;
			expected = XMVector3Rotate(expected, q1);
			expected += t1;

			v = XMVector3Displacement(v, dq);



			Assert::IsTrue(XMVector3NearEqual(v, expected, epsilon), L"Rotation test failed");
		}
	};
}