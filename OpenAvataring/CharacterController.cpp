#include "pch.h"

#include "CharacterController.h"
#include "ClipMetric.h"
#include "PcaCcaMap.h"
#include "EigenExtension.h"
#include "ArmatureTransforms.h"
#include "ArmatureAssignment.h"
#include "StylizedIK.h"
#include "ArmaturePartFeatures.h"

#include "Causality\CharacterObject.h"
#include "Causality\FloatHud.h"
#include "Causality\Settings.h"

#include <filesystem>
#include <random>
#include <tinyxml2.h>
#include <ppl.h>
#include <iterator>

using namespace Causality;
using namespace std;
using namespace Eigen;
using namespace DirectX;
using namespace ArmaturePartFeatures;
using namespace BoneFeatures;
using namespace experimental::filesystem;

//typedef dlib::matrix<double, 0, 1> dlib_vector;

extern Eigen::RowVector3d g_NoiseInterpolation;
const static Eigen::IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
path g_CharacterAnalyzeDir = current_path() / "..\\Assets\\CharacterAnalayze";

#define BEGIN_TO_END(range) range.begin(), range.end()

#ifdef _DEBUG
#define DEBUGOUT(x) std::cout << #x << " = " << x << std::endl
#else
#define DEBUGOUT(x)
#endif

bool ReadGprParamXML(tinyxml2::XMLElement * blockSetting, Eigen::Vector3d &param);
void InitGprXML(tinyxml2::XMLElement * settings, const std::string & blockName, gaussian_process_regression& gpr, gaussian_process_lvm& gplvm);

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

void InitializeExtractor(AllJointRltLclRotLnQuatPcad& ft, const ShrinkedArmature& parts)
{
	// Init 
	ft.InitPcas(parts.size());
	ft.SetDefaultFrame(parts.Armature().bind_frame());
	for (auto part : parts)
	{
		//ft.SetPca(part->Index, part->ChainPcaMatrix, part->ChainPcaMean);
	}
}

class SelfLocalMotionTransform : public ArmatureTransform
{
public:
	const ShrinkedArmature *	pBlockArmature;
	CharacterController*		pController;

	std::vector<std::pair<DirectX::Vector3, DirectX::Vector3>> * pHandles;

	mutable
		Localize<
		EndEffector<InputFeature>>
		inputExtractor;

	mutable
		//Pcad <
		Weighted <
		RelativeDeformation <
		AllJoints < LclRotLnQuatFeature > > > //>
		outputExtractor;

	//mutable MatrixXd m_Xs;

	SelfLocalMotionTransform(CharacterController & controller)
		: pController(&controller), pHandles(nullptr)
	{
		pBlockArmature = &controller.ArmatureParts();
		m_sArmature = &pBlockArmature->Armature();
		m_tArmature = &pBlockArmature->Armature();

		InitializeOutputFeature();
	}

	void InitializeOutputFeature()
	{
		auto& parts = *pBlockArmature;
		auto& ucinfo = pController->GetUnitedClipinfo();

		//outputExtractor.InitPcas(parts.size());
		outputExtractor.SetDefaultFrame(pBlockArmature->Armature().bind_frame());
		outputExtractor.InitializeWeights(parts);

		auto& facade = ucinfo.RcFacade;
		auto cutoff = facade.PcaCutoff();
		for (auto part : parts)
		{
			int pid = part->Index;
			if (part->ActiveActions.size() > 0 || part->SubActiveActions.size() > 0)
			{
				auto &pca = facade.GetPartPca(pid);
				auto d = facade.GetPartPcaDim(pid);
				//outputExtractor.SetPca(pid, pca.components(d), pca.mean());
			}
			else
			{
				int odim = facade.GetPartDimension(pid);
				//outputExtractor.SetPca(pid, MatrixXf::Identity(odim, odim), facade.GetPartMean(pid));
			}
		}
	}

	virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame) override
	{
		const auto& blocks = *pBlockArmature;
		RowVectorXd X, Y;

		for (auto& block : blocks)
		{
			RowVectorXf xf = inputExtractor.Get(*block, source_frame);
			RowVectorXf yf;
			X = xf.cast<double>();

			if (block->ActiveActions.size() > 0)
			{
				auto& sik = pController->GetStylizedIK(block->Index);
				auto& gpr = sik.Gpr();
				gpr.get_ey_on_x(X, &Y);
				yf = Y.cast<float>();
				yf *= block->Wx.cwiseInverse().asDiagonal();

				outputExtractor.Set(*block, target_frame, yf);
			}
			else if (block->SubActiveActions.size() > 0)
			{

			}
		}

		FrameRebuildGlobal(*m_tArmature, target_frame);
	}

	virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, float frame_time) override
	{
		//if (!g_UseVelocity)
		//{
		//	Transform(target_frame, source_frame);
		//	return;
		//}

		const auto& blocks = *pBlockArmature;

		int pvDim = inputExtractor.GetDimension(*blocks[0]);

		RowVectorXd X(g_UseVelocity ? pvDim * 2 : pvDim), Y;

		double semga = 1000;
		RowVectorXf yf;

		std::vector<RowVectorXd> Xabs;
		for (auto& block : blocks)
		{
			//X[0] *= 13;

			if (block->Index > 0 && block->ActiveActions.size() > 0)
			{
				auto& sik = pController->GetStylizedIK(block->Index);
				auto& gpr = sik.Gpr();
				auto& joints = block->Joints;

				RowVectorXf xf = inputExtractor.Get(*block, source_frame);
				RowVectorXf xfl = inputExtractor.Get(*block, last_frame);

				yf = outputExtractor.Get(*block, target_frame);
				auto xyf = inputExtractor.Get(*block, target_frame);
				auto pDecoder = sik.getDecoder();
				auto baseRot = target_frame[block->parent()->Joints.back()->ID].GblRotation;
				//sik.setBaseRotation(baseRot);
				//sik.setChain(block->Joints, target_frame);

				//sik.SetGplvmWeight(block->Wx.cast<double>());

				//std::vector<DirectX::Quaternion, XMAllocator> corrrots(joints.size());
				//std::vector<DirectX::Quaternion, XMAllocator> rots(joints.size());

				//for (int i = 0; i < joints.size(); i++)
				//{
				//	corrrots[i] = target_frame[joints[i]->ID].LclRotation;
				//}

				//(*pDecoder)(rots.data(), yf.cast<double>());
				////outputExtractor.Set(*block, target_frame, yf);
				//auto ep = sik.EndPosition(reinterpret_cast<XMFLOAT4A*>(rots.data()));


				X.segment(0, pvDim) = xf.cast<double>();


				auto Xd = X.segment(0, pvDim);
				auto uXd = gpr.uX.segment(0, pvDim);

				//auto uXv = block->PdGpr.uX.segment<3>(3);
				//Xv = (Xv - uXv).array() * g_NoiseInterpolation.array() + uXv.array();

				Xd = (Xd - uXd).array();

				double varZ = (Xd.array() * (g_NoiseInterpolation.array() - 1.0)).cwiseAbs2().sum();
				// if no noise
				varZ = std::max(varZ, 1e-5);

				Xd = Xd.array() * g_NoiseInterpolation.array() + uXd.array();

				RowVector3d Xld = (xfl.cast<double>() - uXd).array() * g_NoiseInterpolation.array() + uXd.array();

				if (g_UseVelocity)
				{
					auto Xv = X.segment(pvDim, pvDim);
					Xv = (Xd - Xld) / (frame_time * g_FrameTimeScaleFactor);
				}

				xf = X.cast<float>();

				SetVisualizeHandle(block, xf);

				//m_Xs.row(block->Index) = X;
				Xabs.emplace_back(X);

				// Beyesian majarnlize over X
				//size_t detail = 3;
				//MatrixXd Xs(detail*2+1,g_PvDimension), Ys;
				//Xs = gaussian_sample(X, X, detail);

				//VectorXd Pxs = (Xs - X.replicate(detail * 2 + 1, 1)).cwiseAbs2().rowwise().sum();
				//Pxs = (-Pxs.array() / semga).exp();

				//VectorXd Py_xs = block->PdGpr.get_ey_on_x(Xs, &Ys);
				//Py_xs = (-Py_xs.array()).exp() * Pxs.array();
				//Py_xs /= Py_xs.sum();

				//Y = (Ys.array() * Py_xs.replicate(1, Ys.cols()).array()).colwise().sum();

				MatrixXd covObsr(g_PvDimension, g_PvDimension);
				covObsr.setZero();
				covObsr.diagonal() = g_NoiseInterpolation.replicate(1, g_PvDimension / 3).transpose() * varZ;

				//block->PdGpr.get_ey_on_obser_x(X, covObsr, &Y);
				//block->PdGpr.get_expectation(X, &Y);
				//auto yc = yf;
				//yf = Y.cast<float>();
				//yf.array() *= block->Wx.cwiseInverse().array().transpose();

				//block->PdStyleIk.SetHint();
				//sik.setHint(yf.cast<double>());
				//Y = sik.solve(X.transpose(), X.transpose(), baseRot);
				if (!g_UseVelocity)
					Y = sik.apply(X.transpose(), baseRot).cast<double>();
				else
					Y = sik.apply(X.segment(0, pvDim).transpose(), Vector3d(X.segment(pvDim, pvDim).transpose()), baseRot).cast<double>();

				//block->PdStyleIk.SetGoal(X.leftCols<3>());

				//auto scoref = block->PdStyleIk.objective(X, yf.cast<double>());
				//auto scorec = block->PdStyleIk.objective(X, yc.cast<double>());
				//std::cout << "Gpr score : " << scoref << " ; Cannonical score : " << scorec << endl;
				//auto ep = sik.EndPosition(yf.cast<double>());

				//Y = yf.cast<double>();
				outputExtractor.Set(*block, target_frame, Y.cast<float>());
				for (int i = 0; i < block->Joints.size(); i++)
				{
					target_frame[block->Joints[i]->ID].UpdateGlobalData(target_frame[block->Joints[i]->parent()->ID]);
				}

				auto ep2 = target_frame[block->Joints.back()->ID].GblTranslation -
					target_frame[block->Joints[0]->parent()->ID].GblTranslation;

				//break;
			}
		}

		// Fill Xabpv
		if (g_EnableDependentControl && false)
		{
			RowVectorXd Xabpv;
			Xabpv.resize(pController->uXabpv.size());
			int i = 0;
			for (const auto& xab : Xabs)
			{
				auto Yi = Xabpv.segment(i, xab.size());
				Yi = xab;
				i += xab.size();
			}

			auto _x = (Xabpv.cast<float>() - pController->uXabpv) * pController->XabpvT;
			auto _xd = _x.cast<double>().eval();

			for (auto& block : blocks)
			{
				if (block->ActiveActions.size() == 0 && block->SubActiveActions.size() > 0)
				{

					auto& sik = pController->GetStylizedIK(block->Index);
					auto& gpr = sik.Gpr();

					auto lk = gpr.get_ey_on_x(_xd, &Y);

					yf = Y.cast<float>();
					//yf *= block->Wx.cwiseInverse().asDiagonal();

					outputExtractor.Set(*block, target_frame, yf);
				}

			}
		}

		target_frame[0].LclTranslation = source_frame[0].LclTranslation;
		target_frame[0].GblTranslation = source_frame[0].GblTranslation;
		FrameRebuildGlobal(*m_tArmature, target_frame);
	}

	void SetVisualizeHandle(ArmaturePart *const & block, Eigen::RowVectorXf &xf)
	{
		if (pHandles)
		{
			pHandles->at(block->Index).first = Vector3(xf.data());
			if (g_UseVelocity && g_PvDimension == 6)
			{
				pHandles->at(block->Index).second = Vector3(xf.data() + 3);
			}
			else
			{
				pHandles->at(block->Index).second = Vector3::Zero;
			}
		}
	}

	void Visualize();

	//virtual void Transform(_Out_ frame_type& target_frame, _In_ const frame_type& source_frame, _In_ const BoneVelocityFrame& source_velocity, float frame_time) const
	//{
	//	// redirect to pose transform
	//	Transform(target_frame, source_frame);
	//	return;
	//}
};


CharacterController::~CharacterController()
{
}

CharacterController::CharacterController() {
}

void CharacterController::Initialize(CharacterObject & character, const ParamArchive* settings)
{
	IsReady = false;
	CharacterScore = 0;
	CurrentActionIndex = 0;
	SetTargetCharacter(character);
}

const ArmatureTransform & CharacterController::Binding() const { return *m_pBinding; }

ArmatureTransform & CharacterController::Binding() { return *m_pBinding; }

std::mutex & CharacterController::GetBindingMutex()
{
	return m_bindMutex;
}

void CharacterController::SetBinding(std::unique_ptr<ArmatureTransform> && pBinding)
{
	lock_guard<mutex> guard(m_bindMutex);
	m_pBinding = move(pBinding);
}

const ArmatureTransform & CharacterController::SelfBinding() const { return *m_pSelfBinding; }

ArmatureTransform & CharacterController::SelfBinding() { return *m_pSelfBinding; }

const CharacterObject & CharacterController::Character() const { return *m_pCharacter; }

CharacterObject & CharacterController::Character() { return *m_pCharacter; }

const IArmature & CharacterController::Armature() const
{
	return Character().Behavier().Armature();
}

IArmature & CharacterController::Armature()
{
	return Character().Behavier().Armature();
}

const ShrinkedArmature & CharacterController::ArmatureParts() const
{
	return m_charaParts;
}

ShrinkedArmature & CharacterController::ArmatureParts()
{
	return m_charaParts;
}

const std::vector<int>& CharacterController::ActiveParts() const { return m_ActiveParts; }

const std::vector<int>& CharacterController::SubactiveParts() const { return m_SubactiveParts; }

float CharacterController::UpdateTargetCharacter(ArmatureFrameConstView frame, ArmatureFrameConstView lastframe, double deltaTime) const
{
	auto bidning = m_pBinding.get();
	m_updateFrequency = 1.0 / deltaTime;

	if (bidning == nullptr)
		return 0;

	if (try_lock(m_bindMutex))
	{
		std::lock_guard<mutex> guard(m_bindMutex,adopt_lock);
		auto& cframe = m_charaFrame;
		ArmatureFrame f(frame), lf(lastframe);
		auto& sarm = bidning->SourceArmature();
		if (g_IngnoreInputRootRotation)
		{
			RemoveFrameRootTransform(f, sarm);
			RemoveFrameRootTransform(lf, sarm);
		}

		// Linear interpolation between frames
		if (deltaTime > g_MaxiumTimeDelta)
		{
			//! This code pass is buggy!
			ArmatureFrame tf(frame.size()), ltf(lastframe);

			int subStep = (int)ceil(deltaTime / g_MaxiumTimeDelta);
			auto dt = deltaTime / subStep;

			for (int iter = 0; iter < subStep; iter++)
			{
				auto t = (iter + 1)*(1.0 / (subStep));
				FrameLerp(tf, lf, f, t, sarm);

				cout << "sub step [" << iter << ']' << endl;
				bidning->Transform(cframe, tf, ltf, dt);

				ltf = tf;
			}
		}
		else
		{
			bidning->Transform(cframe, f, lf, deltaTime);
		}
		m_pCharacter->MapCurrentFrameForUpdate() = m_charaFrame;
		m_pCharacter->ReleaseCurrentFrameFrorUpdate();
	}

	float l = 100;
	for (auto& bone : frame)
	{
		if (bone.GblTranslation.y < l)
			l = bone.GblTranslation.y;
	}

	using namespace DirectX;

	auto& bone = frame[0];
	//auto pos = frame[0].GblTranslation - MapRefPos + CMapRefPos;

	SychronizeRootDisplacement(bone);

	return 1.0;
}

void CharacterController::SetReferenceSourcePose(const Bone & sourcePose)
{
	auto& chara = *m_pCharacter;
	MapRefPos = sourcePose.GblTranslation;
	LastPos = MapRefPos;
	CMapRefPos = chara.GetPosition();

	MapRefRot = sourcePose.GblRotation;
	CMapRefRot = chara.GetOrientation();

}

void CharacterController::SychronizeRootDisplacement(const Bone & bone) const
{
	auto pos = Character().GetPosition() + bone.GblTranslation - LastPos;

	LastPos = bone.GblTranslation;

	// CrefRot * Yaw(HrefRot^-1 * HRot)
	auto rot = XMQuaternionMultiply(
		XMQuaternionConjugate(XMLoad(MapRefRot)),
		XMLoadA(bone.GblRotation));

	// extract Yaw rotation only, it's a bad hack here
	// code here : Project the rotation to Y-axis, not exactly the Yaw rotation
	rot = XMQuaternionLn(rot);
	rot = XMVectorMultiply(rot, g_XMIdentityR1.v);
	rot = XMQuaternionExp(rot);

	rot = XMQuaternionMultiply(XMLoad(CMapRefRot), rot);


	m_pCharacter->SetPosition(pos);
	//m_pCharacter->SetOrientation(rot);
}

//float CreateControlTransform(CharacterController & controller, const ClipFacade& iclip);

float CharacterController::CreateControlBinding(const ClipFacade& inputClip)
{
	auto info = CreateControlTransform(*this, inputClip);
	if (info.transform != nullptr)
	{
		cout << "Trying to set binding..." << endl;
		SetBinding(move(info.transform));
		CharacterScore = info.likilihood;
		cout << "Finished set binding" << endl;
	}
	return info.likilihood;
}

array_view<std::pair<DirectX::Vector3, DirectX::Vector3>> CharacterController::PvHandles() const
{
	return const_cast<std::vector<std::pair<DirectX::Vector3, DirectX::Vector3>>&>(m_PvHandles);
}

std::vector<std::pair<DirectX::Vector3, DirectX::Vector3>>& CharacterController::PvHandles()
{
	return m_PvHandles;
}

void CharacterController::push_handle(int pid, const std::pair<Vector3, Vector3>& handle) {
	m_PvHandles[pid] = handle;
	m_handelTrajectory[pid].push_back(handle.first);
	if (m_handelTrajectory[pid].size() > m_trajectoryLength)
		m_handelTrajectory[pid].pop_front();
}

CharacterClipinfo & CharacterController::GetClipInfo(const string & name) {
	auto itr = std::find_if(BEGIN_TO_END(m_Clipinfos), [&name](const auto& clip) {
		return clip.ClipName() == name;
	});
	if (itr != m_Clipinfos.end())
	{
		return *itr;
	}
	else
	{
		throw std::out_of_range("given name doesn't exist");
	}
}

template <class DerivedX, class DerivedY, typename Scalar>
void GetVolocity(_In_ const Eigen::DenseBase<DerivedX>& displacement, _Out_ Eigen::DenseBase<DerivedY>& velocity, Scalar frame_time, bool closeLoop = false)
{
	velocity.middleRows(1, CLIP_FRAME_COUNT - 2) = displacement.middleRows(2, CLIP_FRAME_COUNT - 2) - displacement.middleRows(0, CLIP_FRAME_COUNT - 2);

	if (!closeLoop)
	{
		velocity.row(0) = 2 * (displacement.row(1) - displacement.row(0));
		velocity.row(CLIP_FRAME_COUNT - 1) = 2 * (displacement.row(CLIP_FRAME_COUNT - 1) - displacement.row(CLIP_FRAME_COUNT - 2));
	}
	else
	{
		velocity.row(0) = displacement.row(1) - displacement.row(CLIP_FRAME_COUNT - 1);
		velocity.row(CLIP_FRAME_COUNT - 1) = displacement.row(0) - displacement.row(CLIP_FRAME_COUNT - 2);
	}

	velocity /= (2 * frame_time);
}

vector<ArmatureFrame> CreateReinforcedFrames(const BehavierSpace& behavier)
{
	auto& clips = behavier.Clips();
	//auto& parts = behavier.ArmatureParts();
	auto& armature = behavier.Armature();
	auto& dframe = armature.bind_frame();

	float factors[] = { /*0.5f,0.75f,*/1.0f/*,1.25f */ };
	int k = size(factors);
	int n = CLIP_FRAME_COUNT;

	std::vector<int> cyclips;
	cyclips.reserve(clips.size());
	for (int i = 0; i < clips.size(); i++)
	{
		if (clips[i].Cyclic())
			cyclips.push_back(i);
	}

	vector<ArmatureFrame> frames(n * cyclips.size() * k);

	int ci = 0;
	double totalTime = 0;

	//concurrency::parallel_for((int)0, (int)clips.size() * k, [&](int cik) 
	for (int cik = 0; cik < cyclips.size() * k; cik++)
	{
		int ci = cik / k;
		int i = cik % k;

		auto animBuffer = clips[cyclips[ci]].GetFrameBuffer();
		int stidx = cik * n;
		copy_n(animBuffer.begin(), n, frames.begin() + stidx);
		if (fabsf(factors[i] - 1.0f) > 1e-5)
		{
			for (int j = 0; j < n; j++)
			{
				FrameScale(frames[stidx + j], dframe, factors[i]);
				FrameRebuildGlobal(armature, frames[stidx + j]);
			}
		}
	}
	//);

	return frames;
}


void CharacterController::SetTargetCharacter(CharacterObject & chara) {

	using namespace std;
	static std::random_device g_rand;
	static std::mt19937 g_rand_mt(g_rand());

	m_pCharacter = &chara;
	auto& behavier = chara.Behavier();
	auto& armature = chara.Armature();

	//auto sprite = new SpriteObject();
	//m_pCharacter->AddChild(sprite);

	if (m_pBinding)
		m_pBinding->SetTargetArmature(chara.Armature());

	m_characon_add = m_pCharacter->OnChildAdded += [this](SceneObject* _character, SceneObject* _child)
	{
		auto chara = _character->As<CharacterObject>();
		auto& behavier = chara->Behavier();
		auto subchara = _child->As<CharacterObject>();
		if (subchara)
		{
			SubordinateCharacter subordinate;
			subordinate.Character = subchara;
			subchara->SetBehavier(behavier);
			subordinate.PhasePreference = uniform_real<>(0, 1)(g_rand_mt);;
			subordinate.ScalePreference = normal_distribution<>(1, 0.1)(g_rand_mt);
			subordinate.SpeedPreference = normal_distribution<>(1, 0.15)(g_rand_mt);
			subordinate.TimeFilter.Reset();
			subordinate.TimeFilter.SetUpdateFrequency(&m_updateFrequency);
			subordinate.ScaleFilter.SetUpdateFrequency(&m_updateFrequency);
			subordinate.TimeFilter.SetCutoffFrequency(0.3);
			subordinate.TimeFilter.SetCutoffFrequency(0.3);
			this->m_subordinates.push_back(subordinate);
		}
	};

	m_characon_remove = m_pCharacter->OnChildRemoved += [this](SceneObject* _character, SceneObject* _child)
	{
		this->m_subordinates.remove_if([_child](const SubordinateCharacter& sub)
		{ return sub.Character == _child; });
	};


	auto& clips = behavier.Clips();
	auto& parts = m_charaParts;
	parts.SetArmature(armature);

	m_SIKs.resize(parts.size());
	for (size_t i = 0; i < m_SIKs.size(); i++)
	{
		m_SIKs[i].reset(new StylizedChainIK(parts[i]->Joints.size()));
	}

	PotientialFrame = armature.bind_frame();
	m_charaFrame = armature.bind_frame();
	m_PvHandles.resize(armature.size());
	m_trajectoryLength = 30;
	m_handelTrajectory.resize(armature.size());

	//parts.ComputeWeights();
	if (!g_UseJointLengthWeight)
	{
		for (auto& part : parts)
		{
			part->Wx.setOnes();
			part->Wxj.setOnes();
		}
	}

	behavier.UniformQuaternionsBetweenClips();

	for (auto& anim : clips)
	{
		string name = anim.Name;
		std::transform(BEGIN_TO_END(name), name.begin(), std::tolower);

		if (name == "idle" || name == "die" || name == "dead" || name == "death")
			anim.IsCyclic = false;
		else
			anim.IsCyclic = true;
	}

	{
		cout << chara.Name << " Armature Parts" << endl;
		int i = 0;
		for (auto pPart : parts)
		{
			auto& part = *pPart;
			cout << "Part[" << i++ << "] = " << part.Joints;
			if (part.SymetricPair != nullptr)
			{
				cout << " <--> {" << part.SymetricPair->Joints[0] << "...}";
			}
			cout << endl;
		}
	}

	//clips.erase(std::remove_if(BEGIN_TO_END(clips), [](const auto& anim) ->bool {return !anim.IsCyclic;}), clips.end());

	using namespace concurrency;
	vector<task<void>> tasks;
	m_Clipinfos.reserve(clips.size() + 1);
	{
		tasks.emplace_back(create_task([this]() {
			auto& chara = *m_pCharacter;
			auto& behavier = chara.Behavier();
			auto& parts = m_charaParts;

			auto allFrames = CreateReinforcedFrames(behavier);

			//? To-Do Setup proper feature for m_cpxClipinfo
			m_cpxClipinfo.Initialize(parts);
			// set subactive energy to almost zero that make sure all part's pca is caculated
			//m_cpxClipinfo.RcFacade.SetActiveEnergy(g_CharacterActiveEnergy, g_CharacterSubactiveEnergy * 0.01f);
			//m_cpxClipinfo.PvFacade.SetActiveEnergy(g_CharacterActiveEnergy, g_CharacterSubactiveEnergy * 0.01f);
			m_cpxClipinfo.AnalyzeSequence(allFrames, 0, false);
		}));
	}

	for (auto& anim : clips)
	{
		if (!anim.Cyclic())
			continue;

		m_Clipinfos.emplace_back(m_charaParts);
		auto& clipinfo = m_Clipinfos.back();

		clipinfo.SetClipName(anim.Name);

		tasks.emplace_back(create_task([&clipinfo, &anim, &parts]() {
			clipinfo.Initialize(parts);
			auto & frames = anim.GetFrameBuffer();
			clipinfo.AnalyzeSequence(frames, anim.Length().count(), anim.IsCyclic);
		}));
	}

	when_all(tasks.begin(), tasks.end()).then([this]() {

		auto& chara = *m_pCharacter;
		auto& behavier = chara.Behavier();
		auto& clips = behavier.Clips();
		auto& parts = m_charaParts;

		cout << setprecision(4) << setw(6);
		auto& allClipinfo = m_cpxClipinfo;

		tinyxml2::XMLDocument paramdoc;
		tinyxml2::XMLElement* settings = nullptr;
		if (exists(g_CharacterAnalyzeDir))
		{
			create_directory(g_CharacterAnalyzeDir);
		}

		string paramFileName = (g_CharacterAnalyzeDir / (m_pCharacter->Name + ".param.xml")).string();
		stringstream ss;
		ss << "cr_" << CLIP_FRAME_COUNT << "_vel" << (int)g_UseVelocity << "_wj" << g_UseJointLengthWeight;
		string settingName;

		for (auto& cinfo : m_Clipinfos)
			ss << '_' << cinfo.ClipName();

		settingName = ss.str();

		if (g_LoadCharacterModelParameter)
		{
			auto error = paramdoc.LoadFile(paramFileName.c_str());
			tinyxml2::XMLElement* paramStore = nullptr;
			if (error == tinyxml2::XML_SUCCESS)
			{
				paramStore = paramdoc.RootElement();
			}

			if (paramStore == nullptr)
			{
				paramStore = paramdoc.NewElement("param_store");
				paramdoc.InsertFirstChild(paramStore);
			}

			settings = paramStore->FirstChildElement(settingName.c_str());

			if (settings == nullptr)
			{
				settings = paramdoc.NewElement(settingName.c_str());
				paramStore->InsertEndChild(settings);
			}
		}

		float globalEnergyMax = 0;
		for (auto& clipinfo : m_Clipinfos)
		{
			assert(clipinfo.IsReady());
			auto& Epv = clipinfo.PvFacade.GetAllPartsEnergy();
			auto& Erc = clipinfo.RcFacade.GetAllPartsEnergy();
			//auto Eb = Erc.array() * Epv.array();
			auto& Eb = Erc;
			globalEnergyMax = std::max(Eb.maxCoeff(), globalEnergyMax);

			//DEBUGOUT(Eb);
		}

		std::set<int> avtiveSet;
		std::set<int> subactSet;
		for (auto& clipinfo : m_Clipinfos)
		{
			auto& key = clipinfo.ClipName();
			const auto& pvfacade = clipinfo.RcFacade;

			auto& Epv = clipinfo.PvFacade.GetAllPartsEnergy();
			auto& Erc = clipinfo.RcFacade.GetAllPartsEnergy();

			//auto Eb = (Erc.array() * Epv.array()).eval();
			auto& Eb = Erc;
			for (int i = 0; i < Eb.size(); i++)
			{
				if (std::binary_search(BEGIN_TO_END(pvfacade.ActiveParts()),i))
				//if (Eb[i] > g_CharacterActiveEnergy * globalEnergyMax)
				{
					parts[i]->ActiveActions.push_back(key);
					avtiveSet.insert(i);

					// if a part is alreay marked as subactive, promote it
					auto itr = subactSet.find(i);
					if (itr != subactSet.end())
						subactSet.erase(itr);
				}
				else if (avtiveSet.find(i) == avtiveSet.end() && Eb[i] > g_CharacterSubactiveEnergy * globalEnergyMax)
				{
					parts[i]->SubActiveActions.push_back(key);
					subactSet.insert(i);
				}
			}
		}

		// Remove Root from caculation
		avtiveSet.erase(0);
		subactSet.erase(0);
		parts[0]->ActiveActions.clear();
		parts[0]->SubActiveActions.clear();

		m_ActiveParts.assign(BEGIN_TO_END(avtiveSet));
		m_SubactiveParts.assign(BEGIN_TO_END(subactSet));
		vector<int>& activeParts = m_ActiveParts;
		vector<int>& subactParts = m_SubactiveParts;

		for (auto& clipinfo : m_Clipinfos)
		{
			auto& Erc = clipinfo.RcFacade.GetAllPartsEnergy();
		}


		cout << "== Active parts ==" << endl;
		for (auto& ap : activeParts)
			cout << parts[ap]->Joints << endl;
		cout << "== Subactive parts ==" << endl;
		for (auto& ap : subactParts)
			cout << parts[ap]->Joints << endl;
		cout << "== End Parts =" << endl;

		// Active parts Pv s
		MatrixXf Xabpv = GenerateXapv(activeParts);
		int dXabpv = Xabpv.cols();

		parallel_for(0, (int)activeParts.size(), 1, [&, this](int apid)
		{
			InitializeAcvtivePart(*parts[activeParts[apid]], settings);
		}
		);

		if (g_LoadCharacterModelParameter)
		{
			auto error = paramdoc.SaveFile(paramFileName.c_str());
			assert(error == tinyxml2::XML_SUCCESS);
		}

		if (g_EnableDependentControl)
			parallel_for_each(BEGIN_TO_END(subactParts), [&, this](int sapid)
		{
			InitializeSubacvtivePart(*parts[sapid], Xabpv, settings);
		}
		);

		cout << "=================================================================" << endl;

		assert(g_UseStylizedIK);
		{
			//? To-Do, Fix this
			auto pBinding = make_unique<SelfLocalMotionTransform>(*this);
			pBinding->pHandles = &m_PvHandles;
			m_pSelfBinding = move(pBinding);
		}

		if (g_LoadCharacterModelParameter)
		{
			auto error = paramdoc.SaveFile(paramFileName.c_str());
			assert(error == tinyxml2::XML_SUCCESS);
		}

		IsReady = true;
	});
}

MatrixXf CharacterController::GenerateXapv(const std::vector<int> &activeParts)
{
	// pvDim without
	auto& allClipinfo = m_cpxClipinfo;
	auto& pvFacade = allClipinfo.PvFacade;
	int pvDim = pvFacade.GetAllPartDimension();
	auto& parts = m_charaParts;
	assert(pvDim > 0);
	std::vector<int> activeParents = activeParts;
	for (int i = 0; i < activeParts.size(); i++)
		activeParents[i] = parts[activeParts[i]]->parent()->Index;

	MatrixXf Xabpv(allClipinfo.ClipFrames(), size(activeParts) * pvDim);
	MatrixXf Xabparentpv(allClipinfo.ClipFrames(), size(activeParts) * pvDim);

	ArrayXi incX(pvDim);
	incX.setLinSpaced(0, pvDim - 1);

	MatrixXi apMask = VectorXi::Map(activeParts.data(), activeParts.size()).replicate(1, pvDim).transpose();
	MatrixXi appMask = VectorXi::Map(activeParents.data(), activeParents.size()).replicate(1, pvDim).transpose();

	apMask.array() = apMask.array() * pvDim + incX.replicate(1, apMask.cols());
	appMask.array() = appMask.array() * pvDim + incX.replicate(1, appMask.cols());


	auto maskVec = VectorXi::Map(apMask.data(), apMask.size());
	selectCols(pvFacade.GetAllPartsSequence(), maskVec, &Xabpv);
	maskVec = VectorXi::Map(appMask.data(), appMask.size());
	selectCols(pvFacade.GetAllPartsSequence(), maskVec, &Xabparentpv);

	Xabpv -= Xabparentpv;

	Pca<MatrixXf> pcaXabpv(Xabpv);
	int dXabpv = pcaXabpv.reducedRank(g_CharacterPcaCutoff);
	Xabpv = pcaXabpv.coordinates(dXabpv);
	XabpvT = pcaXabpv.components(dXabpv);
	uXabpv = pcaXabpv.mean();

	if (g_EnableDebugLogging)
	{
		ofstream fout(g_CharacterAnalyzeDir / (m_pCharacter->Name + "_Xabpv.pd.csv"));
		fout << Xabpv.format(CSVFormat);

		fout.close();
	}


	return Xabpv;
}

void CharacterController::InitializeAcvtivePart(ArmaturePart & part, tinyxml2::XMLElement * settings)
{
	auto pid = part.Index;
	auto parentid = part.parent()->Index;
	auto& aactions = part.ActiveActions;
	auto& joints = part.Joints;
	auto& allClipinfo = m_cpxClipinfo;

	auto& rcFacade = allClipinfo.RcFacade;
	auto& pvFacade = allClipinfo.PvFacade;

	if (rcFacade.GetPartPcaDim(pid) == -1)
		rcFacade.CaculatePartPcaQr(pid);

	// Prepare the local transform pair
	auto PartPv = pvFacade.GetPartSequence(pid);
	auto ParentPv = pvFacade.GetPartSequence(parentid);
	MatrixXf Pv = PartPv - ParentPv;

	//? To-do Select Active Rows from allClipFacade

	if (g_EnableDebugLogging)
	{
		ofstream fout(g_CharacterAnalyzeDir / (m_pCharacter->Name + "_" + part.Joints[0]->Name + ".pd.csv"));
		fout << Pv.format(CSVFormat);
		fout.close();

		fout.open(g_CharacterAnalyzeDir / (m_pCharacter->Name + "_" + part.Joints[0]->Name + ".x.csv"));
		fout << rcFacade.GetPartSequence(pid).format(CSVFormat);
		fout.close();
	}

	assert(g_UseStylizedIK && "This build is settled on StylizedIK");
	{
		// paramter caching 
		const auto&	partName = joints[0]->Name;
		auto &dframe = Character().Armature().bind_frame();

		auto& sik = *m_SIKs[pid];
		auto& gpr = sik.Gpr();
		auto& gplvm = sik.Gplvm();

		//auto X = rcFacade.GetPartPcadSequence(pid);
		auto X = rcFacade.GetPartSequence(pid);
		gpr.initialize(Pv, X);
		gplvm.initialize(X.cast<double>(), 3);
		InitGprXML(settings, partName, gpr, gplvm);

		auto &pca = rcFacade.GetPartPca(pid);
		auto d = rcFacade.GetPartPcaDim(pid);

		//! PCA Decoder configration
		auto Wjx = part.Wx.cast<double>().eval();

		//auto pDecoder = std::make_unique<RelativeLnQuaternionDecoder>();

		auto pDecoder = std::make_unique<RelativeLnQuaternionPcaDecoder>();
		pDecoder->meanY.setZero(Wjx.size());
		pDecoder->pcaY.setIdentity(Wjx.size(), Wjx.size());
		pDecoder->pcaY.diagonal() = Wjx.cwiseInverse();
		pDecoder->invPcaY.setZero(Wjx.size(), Wjx.size());
		pDecoder->invPcaY.diagonal() = Wjx;

		//pDecoder->meanY = pca.mean().cast<double>() * Wjx.cwiseInverse().asDiagonal();
		//pDecoder->pcaY = pca.components(d).cast<double>();
		//pDecoder->invPcaY = pDecoder->pcaY.transpose();
		//pDecoder->pcaY = Wjx.asDiagonal() * pDecoder->pcaY;
		//pDecoder->invPcaY *= Wjx.cwiseInverse().asDiagonal();

		pDecoder->bases.reserve(joints.size());
		for (auto joint : joints)
		{
			auto jid = joint->ID;
			pDecoder->bases.push_back(dframe[jid].LclRotation);
		}

		sik.setDecoder(move(pDecoder));
		// initialize stylized IK for active chains
		sik.setChain(part.Joints, dframe);

		cout << "Optimal param : " << gpr.get_parameters().transpose() << endl;

	}
}

void CharacterController::InitializeSubacvtivePart(ArmaturePart & part, const Eigen::MatrixXf& Xabpv, tinyxml2::XMLElement * settings)
{
	int pid = part.Index;
	auto& allClipinfo = m_cpxClipinfo;

	auto& rcFacade = allClipinfo.RcFacade;
	//auto& pvFacade = allClipinfo.PvFacade;

	auto& Pv = Xabpv;

	if (rcFacade.GetPartPcaDim(pid) == -1)
		rcFacade.CaculatePartPcaQr(pid);

	auto d = rcFacade.GetPartPcaDim(pid);
	auto X = rcFacade.GetPartPcadSequence(pid);

	if (g_EnableDebugLogging)
	{
		ofstream fout(g_CharacterAnalyzeDir / (m_pCharacter->Name + "_" + part.Joints[0]->Name + ".x.csv"));
		fout << X.format(CSVFormat);
		fout.close();
	}

	if (!g_UseStylizedIK)
	{
		assert(!"this code pass is not valiad. as part.Pd is already Pca-ed here");
	}
	else
	{
		auto& sik = *m_SIKs[pid];
		auto& gpr = sik.Gpr();
		gpr.initialize(Pv, X);

		// paramter caching 
		const auto&	partName = part.Joints[0]->Name;

		//InitGprXML(settings, partName, gpr, sik.Gplvm());
	}
}

template<typename Scalar = double>
Eigen::Matrix<Scalar,-1,-1> readMatrix(const char *filename)
{
	int cols = 0, rows = 0;
	std::vector<Scalar> buff;
	using scalar_iterator_t = std::istream_iterator<Scalar>;

	// Read numbers from file into buffer.
	string line;
	ifstream infile;
	infile.open(filename);
	while (!infile.eof())
	{
		getline(infile, line);

		int temp_cols = 0;
		stringstream stream(line);
		
		scalar_iterator_t sitr(stream);
		std::copy(sitr, scalar_iterator_t(), std::back_inserter(buff));

		if (cols == 0)
			cols = buff.size();

		if (buff.size() > rows * cols)
			rows++;
	}

	infile.close();

	// Populate matrix with numbers.
	Eigen::Matrix<Scalar, -1, -1> result(rows, cols);
	result = Eigen::Map<Matrix<Scalar,-1,-1,Eigen::RowMajor>>(buff.data(), rows, cols);

	return result;
};

bool ReadGplvmParamXML(tinyxml2::XMLElement * blockSetting, Eigen::Vector3d &param, Eigen::MatrixXd& X)
{
	if (blockSetting && blockSetting->Attribute("lvm_alpha") && blockSetting->Attribute("lvm_beta") && blockSetting->Attribute("lvm_gamma"))
	{
		param(0) = blockSetting->DoubleAttribute("lvm_alpha");
		param(1) = blockSetting->DoubleAttribute("lvm_beta");
		param(2) = blockSetting->DoubleAttribute("lvm_gamma");

		auto xsrc = blockSetting->Attribute("lvm_X_src");
		auto path = g_CharacterAnalyzeDir / xsrc;
		X = readMatrix<double>(path.u8string().c_str());

		return true;
	}
	return false;
}

bool ReadGprParamXML(tinyxml2::XMLElement * blockSetting, Eigen::Vector3d &param)
{
	if (blockSetting && blockSetting->Attribute("alpha") && blockSetting->Attribute("beta") && blockSetting->Attribute("gamma"))
	{
		param(0) = blockSetting->DoubleAttribute("alpha");
		param(1) = blockSetting->DoubleAttribute("beta");
		param(2) = blockSetting->DoubleAttribute("gamma");
		return true;
	}
	return false;
}

void InitGprXML(tinyxml2::XMLElement * settings, const std::string & blockName, gaussian_process_regression& gpr, gaussian_process_lvm& gplvm)
{
	gaussian_process_regression::ParamType param;
	gplvm::MatrixType _x;
	bool	paramSetted = false, lvmSetted = false;
	if (g_LoadCharacterModelParameter)
	{
		auto blockSetting = settings->FirstChildElement(blockName.c_str());

		if (paramSetted = ReadGprParamXML(blockSetting, param))
		{
			gpr.set_parameters(param);
		}

		if (lvmSetted = ReadGplvmParamXML(blockSetting, param, _x))
		{
			gplvm.load_model(_x,param);
		}
	}

	if (!paramSetted || !lvmSetted)
	{
		if (!paramSetted)
		{
			gpr.optimze_parameters();
		}
		if (!lvmSetted)
		{
			//gplvm.learn_model(100);
		}

		if (g_LoadCharacterModelParameter)
		{
			auto blockSetting = settings->FirstChildElement(blockName.c_str());

			if (blockSetting == nullptr)
			{
				blockSetting = settings->GetDocument()->NewElement(blockName.c_str());
				settings->InsertEndChild(blockSetting);
			}
			if (!paramSetted)
			{
				param = gpr.get_parameters();
				blockSetting->SetAttribute("alpha", param[0]);
				blockSetting->SetAttribute("beta", param[1]);
				blockSetting->SetAttribute("gamma", param[2]);
			}
			if (!lvmSetted)
			{
				param = gplvm.get_parameters();
				_x = gplvm.latent_coords();

				blockSetting->SetAttribute("lvm_alpha", param[0]);
				blockSetting->SetAttribute("lvm_beta", param[1]);
				blockSetting->SetAttribute("lvm_gamma", param[2]);

				path xcsvsrc = g_CharacterAnalyzeDir / path(blockName + ".csv");
				blockSetting->SetAttribute("lvm_X_src", xcsvsrc.filename().u8string().c_str());
				ofstream of(xcsvsrc);
				of << _x;
				of.close();
			}
		}
	}

	std::cout << "Part gplvm initialized : " << blockName << " = {" << param << '}' << std::endl;

}


void Causality::RemoveFrameRootTransform(ArmatureFrameView frame, const IArmature & armature)
{
	auto& rotBone = frame[armature.root()->ID];
	rotBone.GblRotation = rotBone.LclRotation = Math::Quaternion::Identity;
	rotBone.GblTranslation = rotBone.LclTranslation = Math::Vector3::Zero;
	rotBone.GblScaling = rotBone.LclScaling = Math::Vector3::One;
	FrameRebuildGlobal(armature, frame);
}
