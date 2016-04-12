#include "pch.h"

#include "ArmatureAssignment.h"
#include "ArmatureTransforms.h"
#include "ArmaturePartFeatures.h"
//#define _DEBUG_QAP 1
#include "QudraticAssignment.h"
#include "StylizedIK.h"

#include <limits>
#include <iostream>
#include <filesystem>

#include "EigenExtension.h"

#include <Causality\Settings.h>
#include <Causality\CharacterObject.h>

//#include "Causality\FloatHud.h"


using namespace std;
using namespace Eigen;
using namespace Causality;

typedef array_view<float, -1, -1, -1, -1> array4_view;
typedef array_view<float, -1, -1, -1, -1> array4_const_view;
typedef std::ptrdiff_t index_type;

#ifdef _DEBUG
#define DEBUGOUT(x) std::cout << #x << " = " << x << std::endl
#else
#define DEBUGOUT(x)
#endif

namespace Causality
{
	inline std::ostream& operator<<(std::ostream& os, const Joint& joint)
	{
		os << joint.Name;
		return os;
	}

	inline std::ostream& operator<<(std::ostream& os, const Joint* joint)
	{
		os << joint->Name;
		return os;
	}

	template <class T>
	inline std::ostream& operator<<(std::ostream& os, const std::vector<T> &vec)
	{
		cout << '{';
		for (auto& t : vec)
		{
			cout << t << ", ";
		}
		cout << "\b\b}";
		return os;
	}

	void SetIdentity(Causality::PcaCcaMap & map, const Eigen::Index &rank);
	void SetIdentity(Causality::PcaCcaMap & map, const Eigen::Index &rank)
	{
		map.A.setIdentity(rank, rank);
		map.B.setIdentity(rank, rank);
		map.uX.setZero(rank);
		map.uY.setZero(rank);
		map.uXpca.setZero(rank);
		map.uYpca.setZero(rank);
		map.pcX.setIdentity(rank, rank);
		map.pcY.setIdentity(rank, rank);
		map.useInvB = true;
		map.invB.setIdentity(rank, rank);
	}

	float max_cols_assignment(Eigen::MatrixXf & A, Eigen::MatrixXf & Scor, std::vector<ptrdiff_t> &matching);
	float max_cols_assignment(Eigen::MatrixXf & A, Eigen::MatrixXf & Scor, std::vector<ptrdiff_t> &matching)
	{
		VectorXf XScore(A.rows());
		VectorXi XCount(A.rows());
		XCount.setZero();
		XScore.setZero();
		for (int k = 0; k < A.cols(); k++)
		{
			DenseIndex jx;
			Scor.col(k).maxCoeff(&jx);
			auto score = A(jx, k);
			if (score < g_MatchAccepetanceThreshold) // Reject the match if it's less than a threshold
				matching[k] = -1;
			else
			{
				matching[k] = jx;
				XScore(jx) += score;
				++XCount(jx);
			}
		}
		//XScore.array() /= XCount.array().cast<float>();
		return XScore.sum();
	}

	extern Matrix3f FindIsometricTransformXY(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, float* pError = nullptr);

	Eigen::PermutationMatrix<Eigen::Dynamic> upRotatePermutation(int rows, int rotation)
	{
		Eigen::PermutationMatrix<Eigen::Dynamic> perm(rows);

		for (int i = 0; i < rotation; i++)
		{
			perm.indices()[i] = rows - rotation + i;
		}

		for (int i = 0; i < rows - rotation; i++)
		{
			perm.indices()[rotation + i] = i;
		}
		return perm;
	}

	void TransformHomo(RowVectorXf &xf, const Eigen::MatrixXf& homo)
	{
		xf *= homo.topLeftCorner(homo.rows() - 1, homo.cols() - 1);
		xf += homo.block(homo.rows() - 1, 0, 1, homo.cols() - 1);
	}

	float GetConstraitedRotationFromSinVector(Eigen::Matrix3f &Rot, const Eigen::MatrixXf &covXY, int pivot)
	{
		//RowVector3f angles;
		//for (int i = 0; i < 3; i++)
		//{
		//	float sin = sinRot[i];
		//	if (sin < -1.0f || sin > 1.0f)
		//		angles[i] = 0;
		//	else
		//		angles[i] = asinf(sin);
		//}

		//DirectX::Matrix4x4 rot = DirectX::XMMatrixRotationRollPitchYaw(angles[0], angles[1], angles[2]);
		//for (int i = 0; i < 3; i++)
		//	for (int j = 0; j < 3; j++)
		//		Rot(i, j) = rot(i, j);

		//return;

		// Assumption on one axis rotation

		//DenseIndex pivot = -1;
		//sinRot.cwiseAbs().minCoeff(&pivot);
		float tanX = (covXY(1, 2) - covXY(2, 1)) / (covXY(1, 1) + covXY(2, 2));
		float tanY = (covXY(2, 0) - covXY(0, 2)) / (covXY(0, 0) + covXY(2, 2));
		float tanZ = (covXY(0, 1) - covXY(1, 0)) / (covXY(0, 0) + covXY(1, 1));
		//assert(sin <= 1.0f && sin >= -1.0f && "sin value must within range [0,1]");

		// there is nothing bad about using positive value of cosine, it ensure the angle set in [-pi/2,pi/2]
		float cosX = 1.0f / sqrt(1 + tanX*tanX);
		float sinX = cosX * tanX;
		float cosY = 1.0f / sqrt(1 + tanY*tanY);
		float sinY = cosY * tanY;
		float cosZ = 1.0f / sqrt(1 + tanZ*tanZ);
		float sinZ = cosZ * tanZ;

		sinX = -sinX;
		sinY = -sinY;
		sinZ = -sinZ;
		Rot.setIdentity();

		//! IMPORTANT, Right-Hand 
		switch (pivot)
		{
		case 0:
			Rot(1, 1) = cosX;
			Rot(1, 2) = -sinX;
			Rot(2, 2) = cosX;
			Rot(2, 1) = sinX;
			break;
		case 1:
			Rot(0, 0) = cosY;
			Rot(0, 2) = sinY;
			Rot(2, 2) = cosY;
			Rot(2, 0) = -sinY;
			break;
		case 2:
			Rot(0, 0) = cosZ;
			Rot(0, 1) = -sinZ;
			Rot(1, 1) = cosZ;
			Rot(1, 0) = sinZ;
			break;
		}
		return atanf(tanX);
	}


	Matrix3f FindIsometricTransformXY(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, float* pError)
	{
		assert(X.cols() == Y.cols() && X.rows() == Y.rows() && X.cols() == 3 && "input X,Y dimension disagree");

		auto uX = X.colwise().mean().eval();
		auto uY = Y.colwise().mean().eval();

		//sum(Xi1*Yi1,Xi2*Yi2,Xi3*Yi3)
		MatrixXf covXY = X.transpose() * Y;

		// The one axis rotation matrix
		Matrix3f BestRot;
		float BestScale, bestAng;
		int bestPiv = -1;
		float bestErr = numeric_limits<float>::max();

		for (int pivot = 0; pivot < 3; pivot++)
		{
			Matrix3f Rot;
			float scale = 1.0f;
			float ang = GetConstraitedRotationFromSinVector(Rot, covXY, pivot);
			// the isometric scale factor
			scale = ((X * Rot).array()*Y.array()).sum() / X.cwiseAbs2().sum();
			float err = (X * scale * Rot - Y).cwiseAbs2().sum();
			if (err < bestErr)
			{
				bestPiv = pivot;
				BestRot = Rot;
				BestScale = scale;
				bestErr = err;
				bestAng = ang;
			}
		}

		if (g_EnableDebugLogging >= 1)
		{
			if (bestPiv == -1)
			{
				cout << "[!] Error , Failed to find isometric transform about control handle" << endl;
			}
			else
			{
				static char xyz[] = "XYZ";
				cout << "Isometric transform found : Scale [" << BestScale << "] , Rotation along axis [" << xyz[bestPiv] << "] for " << bestAng / DirectX::XM_PI << "pi , Error = " << bestErr << endl;
			}
		}

		if (pError != nullptr)
		{
			*pError = bestErr;
		}
		return BestScale * BestRot;
	}

	// Return the average re-construction error in squared-distance form
	float FindPartToPartTransform(_Inout_ P2PTransform& transform, const ClipFacade& iclip, const ClipFacade& cclip, size_t phi)
	{
		int ju = transform.SrcIdx;
		int jc = transform.DstIdx;

		// Up-rotate X to phi
		auto up = iclip.ArmatureParts()[ju]->parent();
		int jup = up ? up->Index : -1;
		auto cp = cclip.ArmatureParts()[jc]->parent();
		int jcp = cp ? cp->Index : -1;

		bool isSinglePose = iclip.ClipFrames() == 1;
		//std::cout << isSinglePose << endl;


		MatrixXf rawX;

		int nx = iclip.ClipFrames();
		int ny = cclip.ClipFrames();

		auto rotX = upRotatePermutation(nx, phi);


		auto rawY = (cclip.GetPartSequence(jc) - cclip.GetPartSequence(jcp)).eval();

		if (isSinglePose)
		{
			assert(isSinglePose);
			if (g_EnableDebugLogging > 1)
				std::cout <<"signle frame binding : " << isSinglePose << endl;
			rawX = (iclip.GetPartSequence(ju) - iclip.GetPartSequence(jup)).replicate(rawY.rows(), 1);
		}
		else {
			assert(!isSinglePose);
			if (g_EnableDebugLogging > 1)
				std::cout << "periodic motion binding : " << isSinglePose << endl;
			rawX = (rotX * (iclip.GetPartSequence(ju) - iclip.GetPartSequence(jup))).eval();
		}

		//assert(rawX.rows() == rawY.rows() && rawY.rows() == T);

		float err = 0;
		if (g_PartAssignmentTransform == PAT_CCA)
		{
			assert(!"PAT_CCA is currently buggy!");
			PcaCcaMap map;
			map.CreateFrom(rawX, rawY, iclip.PcaCutoff(), cclip.PcaCutoff());
			transform.HomoMatrix = map.TransformMatrix();
		}
		else if (g_PartAssignmentTransform == PAT_OneAxisRotation)
		{
			//RowVectorXf alpha = (rawY.cwiseAbs2().colwise().sum().array() / rawX.cwiseAbs2().colwise().sum().array()).cwiseSqrt();
			//float err = (rawY - rawX * alpha.asDiagonal()).cwiseAbs2().sum();

			auto Transf = FindIsometricTransformXY(rawX, rawY, &err);
			auto rank = rawX.cols();

			transform.HomoMatrix.setIdentity(4, 4);
			transform.HomoMatrix.topLeftCorner(3, 3) = Transf;
		}
		else if (g_PartAssignmentTransform == PAT_AnisometricScale)
		{
			RowVectorXf alpha = (rawY.cwiseAbs2().colwise().sum().array() / rawX.cwiseAbs2().colwise().sum().array()).cwiseSqrt();
			err = (rawY - rawY * alpha.asDiagonal()).cwiseAbs2().sum();

			transform.HomoMatrix.setIdentity(4, 4);
			transform.HomoMatrix.topLeftCorner(3, 3) = alpha.asDiagonal();
		}
		else if (g_PartAssignmentTransform == PAT_RST)
		{
			RowVectorXf uX = rawX.colwise().mean();
			RowVectorXf uY = rawY.colwise().mean();
			assert(uX.size() == uY.size());
			MatrixXf _X = rawX - uX.replicate(rawX.rows(), 1);
			MatrixXf _Y = rawY - uY.replicate(rawY.rows(), 1);

			//float unis = sqrtf(uY.stableNorm() / uY.stableNorm());
			float unis = 1.0f;
			RowVectorXf alpha;

			if (isSinglePose) {
				if (g_EnableDebugLogging > 1)
					std::cout << "signle frame binding : " << isSinglePose << endl;
				unis = uY.norm() / uX.norm();
				alpha = uY.array() / uX.array();
			}
			else
			{
				if (g_EnableDebugLogging > 1)
					std::cout << "signle frame binding : " << isSinglePose << endl;
				unis = sqrtf(rawY.cwiseAbs2().sum() / rawX.cwiseAbs2().sum());
				alpha = (_Y.cwiseAbs2().colwise().sum().array() / _X.cwiseAbs2().colwise().sum().array()).cwiseSqrt();
			}

			err = (_Y - _X * alpha.asDiagonal()).cwiseAbs2().sum();
			alpha = alpha.cwiseMax(g_HandleTransformAnisometricMin * unis).cwiseMin(g_HandleTransformAnisometricMax * unis);
			std::cout << "Global scale = " << unis << " Anisometric scale = " << alpha << std::endl;

			//alpha[2] = -alpha[2];
			//float flipErr = (_Y - _X * alpha.asDiagonal()).cwiseAbs2().sum();
			//if (err < flipErr)
			//{
			//	alpha[2] = -alpha[2];
			//}

			auto rank = rawX.cols();

			auto& homo = transform.HomoMatrix;
			homo.setIdentity(uX.size() + 1, uY.size() + 1);
			homo.topLeftCorner(uX.size(), uY.size()) = alpha.asDiagonal();
			homo.block(uX.size(), 0, 1, uY.size()) = -uX*alpha.asDiagonal() + uY;

			//uX = _X.row(0);
			//uY = uX * alpha.asDiagonal() + uY;
			//uX = rawX.row(0);
			//TransformHomo(uX, homo);

			//uY = uX;
		}
		return err;
	}

	bool is_symetric(const ArmaturePart& lhs, const ArmaturePart& rhs)
	{
		return
			(lhs.SymetricPair != nullptr && lhs.SymetricPair->Index == rhs.Index)
			|| (rhs.SymetricPair != nullptr && rhs.SymetricPair->Index == lhs.Index);
	}

	// helper functions
	void CaculateQuadraticDistanceMatrix(array4_view &C, _In_ const MatrixXf& A, const ClipFacade& iclip, const ClipFacade& cclip, const MatrixXi& iance, const MatrixXi& cance, const VectorXf& Eub, const VectorXf& Ecb)
	{
		auto& Juk = iclip.ActiveParts();
		auto& Jck = cclip.ActiveParts();
		//const std::vector<int> &Juk, const std::vector<int> &Jck, const Eigen::Array<Eigen::RowVector3f, -1, -1> &XpMean, const Eigen::Array<Eigen::Matrix3f, -1, -1> &XpCov, const Causality::CharacterController & controller);

		auto& cparts = cclip.ArmatureParts();
		auto& sparts = iclip.ArmatureParts();

		for (index_type i = 0; i < Juk.size(); i++)
		{
			for (index_type j = i + 1; j < Juk.size(); j++)
			{
				for (index_type si = 0; si < Jck.size(); si++)
				{
					for (index_type sj = si + 1; sj < Jck.size(); sj++)
					{
						auto xu = iclip.GetPartsDifferenceMean(Juk[i], Juk[j]);
						auto xc = cclip.GetPartsDifferenceMean(Jck[si], Jck[sj]);
						auto cu = iclip.GetPartsDifferenceCovarience(Juk[i], Juk[j]);
						auto cc = cclip.GetPartsDifferenceCovarience(Jck[si], Jck[sj]);

						auto& sparti = *sparts[Juk[i]];
						auto& spartj = *sparts[Juk[j]];
						auto& cparti = *cparts[Jck[si]];
						auto& cpartj = *cparts[Jck[sj]];

						float val = 0;
						// both are non-zero
						if (xu.norm() > 0.01f && xc.norm() > 0.01f)
						{

							//auto edim = (-cu.diagonal() - cc.diagonal()).array().exp().eval();
							//RowVector3f _x = xu.array() * xc.array();
							//_x /= xu.norm() * xc.norm();
							val = xu.dot(xc) / (xu.norm() * xc.norm());
							assert(fabsf(val) <= 1.01f);
							//val = (_x.array() /** edim.transpose()*/).sum();
							//val *=  /** sqrt(sqrt(Eub[i]*Eub[j]*Ecb[si]*Ecb[sj]))*/;
							val *= g_PartAssignmentQuadraticTermWeight * sqrt(A(i, si) * A(j, sj));

							float extra = .0f;

							// structrual bounus
							bool csym = is_symetric(cparti, cpartj);
							bool ssym = false;
							if (csym)
							{
								ssym = is_symetric(sparti, spartj);
								if (ssym)
								{
									extra = g_StructrualSymtricBonus * sqrt(A(i, si) * A(j, sj));
									assert(extra >= .0f);
								}
								else
								{
									extra = g_StructrualDisSymtricPenalty;
									assert(extra <= .0f);
								}
							}

							float dir_extra = .0f;
							float anti_extra = .0f;

							int canc = cance(si, sj);
							int sanc = iance(i, j);
							if (cance(si, sj) != 0)
							{
								int anceType = iance(i, j) * cance(si, sj);

								// Prevent unrelated parts mapps to an related structure
								if (anceType == 0)
									extra += g_StructtualAntiAncestorPenalty;
								else if (anceType == -1)
									dir_extra = g_StructtualAntiAncestorPenalty;
								else
									anti_extra = g_StructtualAntiAncestorPenalty;
							}


							if (g_EnableDebugLogging >= 2)
							{
								std::cout << "C(" << i << ',' << j << ',' << si << ',' << sj << ")= "
									<< val << " ; extra=" << extra << " dir=" << dir_extra << " anti=" << anti_extra <<
									" ; c-sym=" << csym << " s-sym=" << ssym <<
									" ; c-anc=" << canc << " s-anc=" << sanc << endl;
							}

							C(i, j, si, sj) = val + extra + dir_extra;
							C(j, i, sj, si) = val + extra + dir_extra;
							C(i, j, sj, si) = -val + extra + anti_extra;
							C(j, i, si, sj) = -val + extra + anti_extra;


						} // one of them is zero
						else if (xu.norm() + xc.norm() > 0.01f)
						{
							val = -g_PartAssignmentQuadraticTermWeight;
							C(i, j, si, sj) = val;
							C(j, i, sj, si) = val;
							C(i, j, sj, si) = val;
							C(j, i, si, sj) = val;
						}
						else // both zero
						{
							C(i, j, si, sj) = 0;
							C(j, i, sj, si) = 0;
							C(i, j, sj, si) = 0;
							C(j, i, si, sj) = 0;
						}

					}
				}
			}
		}

		//DEBUGOUT(C);
	}

}

void ConvertCovToPca(Matrix3f& cov, float norm)
{
	cov /= norm;
	auto svd = cov.jacobiSvd(Eigen::ComputeFullV);
	cov = svd.singularValues().asDiagonal() * svd.matrixV();
}

VectorXi CaculateConditionalSymetricVector(const ShrinkedArmature& parts, const std::vector<int> &activeParts)
{
	VectorXi symType(activeParts.size());
	symType.setConstant(0);
	for (int i = 0; i < activeParts.size(); i++)
	{
		auto part = parts[activeParts[i]];
		auto mid = part->SymetricPair ? part->SymetricPair->Index : -1;
		if (mid != -1)
		{
			symType[i] = std::find(activeParts.begin(), activeParts.end(), mid) != activeParts.end();
		}
		else
			symType[i] = 0;
	}
	return symType;
}

namespace stdx
{
	template <class T, bool ownership>
	bool is_ancester(const tree_node<T, ownership>* node, const tree_node<T, ownership> *ancester)
	{
		if (!ancester->has_child()) return false;

		do 
			node = node->parent();
		while (node != nullptr && node != ancester);

		return node == ancester;
	}
}

MatrixXi CaculateAncesterMatrix(const ShrinkedArmature& parts, const std::vector<int> &activeParts)
{
	int m = activeParts.size();
	MatrixXi ancMap(m, m);
	ancMap.setZero();

	for (int i = 0; i < activeParts.size(); i++)
	{
		auto& part = *parts[activeParts[i]];
		for (int j = i + 1; j <  activeParts.size(); j++)
		{
			int ance = stdx::is_ancester(parts[activeParts[j]], parts[activeParts[i]]);
			ancMap(i, j) = ance;
			if (ance)
				ancMap(j, i) = -1;
		}
	}
	return ancMap;
}

inline static float sqr(float v)
{
	return v*v;
}

CtrlTransformInfo Causality::CreateControlTransform(CharacterController & controller, const ClipFacade& iclip, const string& character_actionName)
{
	assert(controller.IsReady && iclip.IsReady());

	const size_t pvDim = iclip.GetAllPartDimension();
	// alias setup
	auto& character = controller.Character();
	auto& charaParts = controller.ArmatureParts();
	auto& userParts = iclip.ArmatureParts();
	size_t Jc = charaParts.size();
	auto& clips = character.Behavier().Clips();
	auto& clipinfos = controller.GetClipInfos();

	int framesCount = iclip.ClipFrames();
	bool isSingleFrame = framesCount == 1;

	//controller.CharacterScore = numeric_limits<float>::min();
	//auto& anim = character.Behavier()["walk"];

	//if (character.CurrentAction() == nullptr)
	//	return 0.0f;

	//auto panim = character.CurrentAction();
	//if (panim == nullptr)
	//	panim = &character.Behavier()[character_actionName];
	//auto& anim = *panim;

	auto& anim = character.Behavier()[character_actionName];

	int T = iclip.ClipFrames(); //? /2 Maybe?
	const std::vector<int> &Juk = iclip.ActiveParts();
	std::vector<int> Juk3(Juk.size() * pvDim);
	for (int i = 0; i < Juk.size(); i++)
		for (int j = 0; j < pvDim; j++)
			Juk3[i * pvDim + j] = Juk[i] * pvDim + j;

	int Ti = g_PhaseMatchingInterval;
	int Ts = T / Ti + 1;


	VectorXf Eub(Juk.size());
	selectRows(iclip.GetAllPartsEnergy().transpose(), Juk, &Eub);
	NormalizeEnergyVector(Eub);
	// Player Perceptive vector mean normalized
	MatrixXf Xpvnm(pvDim, Juk.size());
	std::vector<Matrix3f> Xpvcov(Juk.size());

	MatrixXf Xpvseq(iclip.ClipFrames(), Juk.size() * pvDim);
	selectCols(iclip.GetAllPartsSequence(), Juk3, &Xpvseq);
	RowVectorXf xpvrow = Xpvseq.colwise().sum();

	VectorXi XsymType = CaculateConditionalSymetricVector(userParts, Juk);
	MatrixXi XAncType = CaculateAncesterMatrix(userParts, Juk);

	//selectCols(reshape(iclip.GetAllPartsMean(), pvDim, -1), Juk, &Xpvnm);
	for (int i = 0; i < Juk.size(); i++)
	{
		int pid = userParts[Juk[i]]->parent()->Index;
		Xpvnm.col(i) = iclip.GetPartsDifferenceMean(Juk[i], pid).transpose();
		auto norm = Xpvnm.col(i).norm();
		Xpvnm.col(i) /= norm;

		Xpvcov[i] = iclip.GetPartsDifferenceCovarience(Juk[i], pid);
		auto sn = Xpvcov[i].diagonal().sum();

		if (abs(sn) < 0.01f)
			Xpvcov[i].setZero();
		else
			Xpvcov[i] /= Xpvcov[i].diagonal().sum();
		//Xpvcov[i] = Xpvcov[i].cwiseSqrt();
		//ConvertCovToPca(Xpvcov[i],norm);

	}
	//Xpvnm = reshape(xpvrow, Juk.size(), pvDim).transpose().colwise().normalized();
	//Xpvnm.colwise().normalize();

	std::vector<unique_ptr<PartilizedTransformer>> clipTransforms;
	clipTransforms.reserve(clips.size());
	Eigen::VectorXf clipTransformScores(clips.size());
	clipTransformScores.setZero();
	//for (auto& cclip : controller.GetClipInfos())	//? <= 5 animation per character
	{
		auto& cclip = controller.GetClipInfo(anim.Name);
		auto& cpv = cclip.PvFacade;

		// Independent Active blocks only
		//std::vector<int> Jck;
		//Jck.reserve(cpv.ActiveParts().size());
		//for (auto cj : cpv.ActiveParts())
		//{
		//	if (std::binary_search(controller.ActiveParts().begin(), controller.ActiveParts().end(), cj))
		//	{
		//		Jck.push_back(cj);
		//	}
		//}
		const auto &Jck = cpv.ActiveParts();

		VectorXi CsymType = CaculateConditionalSymetricVector(charaParts, Jck);
		MatrixXi CAncType = CaculateAncesterMatrix(charaParts, Jck);

		//const auto &Jck = controller.ActiveParts();

		//std::vector<int> Jck3(Jck.size() * pvDim);
		//for (int i = 0; i < Jck.size(); i++)
		//	for (int j = 0; j < pvDim; j++)
		//		Jck3[i * pvDim + j] = Jck[i] * pvDim + j;


		// Ecb, Energy of Character Active Parts
		VectorXf Ecb(Jck.size());
		// Ecb3, Directional Energy of Character Active Parts
		MatrixXf Ecb3(pvDim, Jck.size());

		selectRows(cpv.GetAllPartsEnergy().transpose(), Jck, &Ecb);
		Ecb = 1.0f + Ecb.array() / Ecb.maxCoeff();

		//selectCols(cclip.Eb3, Jck, &Ecb3);
		for (size_t i = 0; i < Jck.size(); i++)
			Ecb3.col(i) = cpv.GetPartDimEnergy(Jck[i]);
		Ecb3.colwise().normalize();

		// Character Perceptive vector mean normalized
		MatrixXf Cpvnm(pvDim, Jck.size());
		std::vector<Matrix3f> Cpvcov(Jck.size());
		VectorXi Cpvdim(Jck.size());
		//RowVectorXf cpvnmrow(pvDim* Jck.size());
		//selectCols(cpv.GetAllPartsMean(), Jck3, &cpvnmrow);
		//Cpvnm = reshape(cpvnmrow, Jck.size(), pvDim).transpose().colwise().normalized();
		for (int i = 0; i < Jck.size(); i++)
		{
			int pid = 0;
			int jid = Jck[i];
			if (charaParts[Jck[i]]->parent() != nullptr)
				charaParts[Jck[i]]->parent()->Index;
			Cpvnm.col(i) = cpv.GetPartsDifferenceMean(Jck[i], charaParts[Jck[i]]->parent()->Index).transpose();
			auto norm = Cpvnm.col(i).norm();
			Cpvnm.col(i) /= norm;

			Cpvcov[i] = cpv.GetPartsDifferenceCovarience(Jck[i], pid);
			auto sn = Cpvcov[i].diagonal().sum();
			if (abs(sn) < 0.01f)
				Cpvcov[i].setZero();
			else
				Cpvcov[i] /= Cpvcov[i].diagonal().sum();
			//Cpvcov[i] = Cpvcov[i].cwiseSqrt();
			//ConvertCovToPca(Xpvcov[i], norm);

			//Cpvdim[i] = svals[0] / svals.norm() > 0.95f ? 1 : // 1-D Trajectory
			//			svals[2] / svals.norm() < 0.05 ? 2 :  // 2-D Trajectory
			//			3; // 3-D Spherical Trajectory

			//Cpvcov[i] = svd.singularValues().asDiagonal() * svd.matrixV();
		}

		if (g_EnableDebugLogging >= 1)
		{
			cout << "=============================================" << endl;
			cout << "Bodypart assignment for " << character.Name << " : " << anim.Name << endl;
			if (isSingleFrame)
				cout << "!!!!!!!!Single-frame-binding!!!!!!!!!!!" << endl;
			cout << "*********************************************" << endl;
			cout << "Human Skeleton ArmatureParts : " << endl;

			int iid = 0; // For enumeration over SymTypes
			for (auto i : Juk)
			{
				const auto& blX = *userParts[i];
				cout << "Part[" << i << "] (" << (XsymType[iid++]?'E':'S') << ") = " << blX.Joints << endl;
			}

			cout << "*********************************************" << endl;
			cout << "Character " << character.Name << "'s Skeleton ArmatureParts : " << endl;
			
			iid = 0;
			for (auto& i : Jck)
			{
				const auto& blY = *charaParts[i];
				cout << "Part[" << i << "] (" << (CsymType[iid++] ? 'E' : 'S') <<") = " << blY.Joints << endl;
			}
			cout << "Caculating Quadratic Assignment Problem ... " << endl;
		}

		//Cpvnm.colwise().normalize();

		//MatrixXf Cpvseq(cpv.ClipFrames(), Jck.size() * pvDim);
		//selectCols(cpv.GetAllPartsSequence(), Jck3, &Cpvseq);


		// Memery allocation
		auto CoRSize = Juk.size() + Jck.size();

		MatrixXf A(Juk.size(), Jck.size());

		// Caculate Bipetral Matching Distance Matrix A
		// Eb3 is ensitially varience matrix here
		for (int i = 0; i < Juk.size(); i++)
		{
			for (int j = 0; j < Jck.size(); j++)
			{
				float vgeom = ((Xpvnm.col(i) - Cpvnm.col(j)).array() * Ecb3.col(j).array()).matrix().norm();
				float vdync = (Xpvcov[i].diagonal().cwiseSqrt() - Cpvcov[i].diagonal().cwiseSqrt()).norm();
				A(i, j) = sqrt(sqr(vgeom) + sqr(g_DynamicTermWeight * vdync));
			}
		}

		// Anisometric Gaussian kernal here
		A.array() = (-(A.array() / (DirectX::XM_PI / 6)).cwiseAbs2()).exp();

		A = Eub.cwiseSqrt().asDiagonal() * A * Ecb.cwiseSqrt().asDiagonal();

		if (g_EnableDebugLogging >= 1)
		{
			cout << " A = " << endl;
			cout << A << endl;
		}

		//A.noalias() = Xsp.transpose() * Csp;

		//_ASSERTE( _CrtCheckMemory( ) );

		typedef array4_view::size_type size_type;
		size_type cu = Juk.size(), cc = Jck.size();
		std::vector<float> Cdata(cu*cu*cc*cc, .0f);
		array4_view C(Cdata.data(), array4_view::bounds_type({ cu, cu, cc, cc }));

		//_ASSERTE(_CrtCheckMemory());
		CaculateQuadraticDistanceMatrix(C, A, iclip, cpv, XAncType, CAncType,Eub,Ecb);

		for (int i = 0; i < Juk.size(); i++)
		{
			for (int j = 0; j < Jck.size(); j++)
			{
				// Bounus for Singular to Singular Assignment
				if (!(XsymType[i] || CsymType[j]))
					A(i, j) *= (1.0f + g_StructrualSymtricBonus);
			}
		}

		//cout << C << endl;

		vector<index_type> matching(Juk.size());

		//float score = 0;
		//for (int i = 0; i < Jck.size(); i++)
		//{
		//	Index idx;
		//	matching[i] = -1;
		//	score += A.col(i).maxCoeff(&idx);
		//	matching[i] = idx;
		//}

		auto naiveMatching = max_weight_bipartite_matching(A);
		auto naiveScore = matching_cost(A, naiveMatching);

		float score = max_quadratic_assignment(A, array4_const_view(C), gsl::span<index_type>(matching));

		float maxScore = score;

		//matching.clear();
		//matching.shrink_to_fit();


#pragma region Display Debug Armature Parts Info
		if (g_EnableDebugLogging >= 1)
		{
			cout << "__________ Parts Assignment __________" << endl;
			cout << "Scores : " << maxScore << endl;
			for (int i = 0; i < matching.size(); i++)
			{
				if (matching[i] < 0) continue;
				int ju = Juk[i], jc = Jck[matching[i]];
				if (ju >= 0 && jc >= 0)
				{
					cout << userParts[ju]->Joints << " ==> " << charaParts[jc]->Joints << endl;
				}
			}
			cout << "__________ Fin __________" << endl;
		}
#pragma endregion

		VectorXi maxPhi(matching.size());
		maxPhi.setZero();

		//_ASSERTE( _CrtCheckMemory( ) );
		if (!isSingleFrame)
		{
			MatrixXf corrlations(Ts, matching.size());
			corrlations.setZero();
			Cca<float> cca;

			for (int i = 0; i < matching.size(); i++)
			{
				if (matching[i] < 0) continue;
				int ju = Juk[i], jc = Jck[matching[i]];
				for (int phi = 0; phi < T; phi += Ti)
				{
					if (ju >= 0 && jc >= 0)
					{
						cca.computeFromQr(iclip.GetPartPcadQrView(ju), cpv.GetPartPcadQrView(jc), false, phi);
						corrlations(phi / Ti, i) = cca.correlaltions().minCoeff();
					}
					else
						corrlations(phi / Ti, i) = 0;
				}
			}

			//float sumCor = corrlations.rowwise().sum().maxCoeff(&maxPhi);

			int misAlign = Ts / 5;
			//? maybe other reduce function like min?
			//! We should allowed a window of range for phi matching among different parts
			corrlations.conservativeResize(Ts + misAlign, corrlations.cols());
			corrlations.bottomRows(misAlign) = corrlations.topRows(misAlign);
			{
				int mPhis = 0;
				float scorePhase = numeric_limits<float>::min();;
				for (int i = 0; i < Ts; i++)
				{
					float score = corrlations.middleRows(i, misAlign).colwise().maxCoeff().sum();
					if (score > scorePhase)
					{
						mPhis = i;
						scorePhase = score;
					}
				}

				for (int i = 0; i < corrlations.cols(); i++)
				{
					corrlations.middleRows(mPhis, misAlign).col(i).maxCoeff(&maxPhi[i]);
					maxPhi[i] += mPhis;
					if (maxPhi[i] >= Ts) maxPhi[i] -= Ts;
					maxPhi[i] *= Ti;
				}

				// Combine the score from qudratic assignment with phase matching
				maxScore = maxScore /** scorePhase*/;

				//_ASSERTE( _CrtCheckMemory( ) );

			}
		}


		// Transform pair for active parts
		std::vector<P2PTransform> partTransforms;
		partTransforms.reserve(matching.size());

		//_ASSERTE( _CrtCheckMemory( ) );

		double partAssignError = 0;
		for (int i = 0; i < matching.size(); i++)
		{
			if (matching[i] < 0) continue;
			int ju = Juk[i], jc = Jck[matching[i]];
			if (ju >= 0 && jc >= 0)
			{
				partTransforms.emplace_back();
				auto &partTra = partTransforms.back();
				partTra.DstIdx = jc;
				partTra.SrcIdx = ju;

				auto err = FindPartToPartTransform(partTra, iclip, cpv, maxPhi[i]);
				partAssignError += err / cpv.ClipFrames();
			}
		}

		//_ASSERTE(_CrtIsValidHeapPointer(matching.data()));

		//matching.clear();
		//matching.shrink_to_fit();

		//_ASSERTE( _CrtCheckMemory() );

		auto pTransformer = new PartilizedTransformer(userParts, controller);
		pTransformer->ActiveParts = move(partTransforms);

		pTransformer->SetupTrackers(
			partAssignError,
			g_TrackerSubStep,
			g_TrackerVtProgationStep,
			g_TrackerScaleProgationStep,
			g_TrackerStDevVt,
			g_TrackerStDevScale,
			g_TrackerTimeSubdivide,
			g_TrackerVtSubdivide,
			g_TrackerSclSubdivide);

		pTransformer->EnableTracker(anim.Name);

		//_ASSERTE( _CrtCheckMemory( ) );

		pTransformer->GenerateDrivenAccesseryControl();

		clipTransformScores[clipTransforms.size()] = maxScore;
		clipTransforms.emplace_back(std::move(pTransformer));

	} // Animation clip scope

	DenseIndex maxClipIdx = -1;
	float maxScore = clipTransformScores.segment(0, clipTransforms.size()).maxCoeff(&maxClipIdx);

	cout << maxScore << endl;

	CtrlTransformInfo candy;
	candy.likilihood = 0;
	candy.clipname = anim.Name;
	if (maxClipIdx >= 0 
		/*&& (&controller.Binding() == nullptr || maxScore > controller.CharacterScore * 1.2)*/)
	{
		candy.likilihood = maxScore;
		candy.transform = move(clipTransforms[maxClipIdx]);
		//cout << "Trying to set binding..." << endl;
		//controller.SetBinding(move(clipTransforms[maxClipIdx]));
		//controller.CharacterScore = maxScore;
		//cout << "Finished set binding" << endl;
	}

	return candy;

	//if (g_EnableDependentControl)
	//{
	//	for (auto& pBlock : charaParts)
	//	{
	//		auto& block = *pBlock;
	//		//if (block.Index == 0)
	//		//	continue;
	//		if (block.ActiveActionCount == 0 && block.SubActiveActionCount > 0)
	//		{
	//			pBinding->Maps.emplace_back(block.PdCca);
	//		}
	//	}
	//}

}

void Causality::NormalizeEnergyVector(Eigen::VectorXf &Eub)
{
	Eub = .5f + 0.5 * Eub.array() / Eub.maxCoeff();
}
