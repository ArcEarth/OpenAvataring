#pragma once
#include <vector>
#include <map>
#include "Common\Filter.h"
#include "Causality\Animations.h"

#include "CCA.h"
#include "RegressionModel.h"
#include "PcaCcaMap.h"
#include "ArmatureParts.h"
#include "BoneFeatures.h"
#include "ArmaturePartFeatures.h"
#include "CharacterActionTracker.h"

namespace Causality
{
	class MatrixVisualizer;

	struct TransformPair
	{
		int Jx, Jy;
		std::unique_ptr<IRegression> pRegression;
	};

	class CharacterController;

	namespace ArmaturePartFeatures
	{
		class EndEffectorGblPosQuadratized : public IArmaturePartFeature
		{
		public:
			EndEffectorGblPosQuadratized();

			int GetDimension(_In_ const ArmaturePart& block) const override;

			typedef BoneFeatures::QuadraticGblPosFeature BoneFeatureType;
			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override;

			// Inherited via IArmaturePartFeature
			virtual void Set(const ArmaturePart & block, ArmatureFrameView frame, const Eigen::RowVectorXf & feature) override;
		};

		class AllJointRltLclRotLnQuatPcad : public Pcad<RelativeDeformation<AllJoints<BoneFeatures::LclRotLnQuatFeature>>>
		{
		public:
			typedef Pcad<RelativeDeformation<AllJoints<BoneFeatures::LclRotLnQuatFeature>>> BaseType;


		};
	}

	class BlockizedArmatureTransform : public ArmatureTransform
	{
	protected:
		const ShrinkedArmature *m_sParts, *m_cParts;
	public:

		BlockizedArmatureTransform();

		BlockizedArmatureTransform(const ShrinkedArmature * pSourceBlock, const ShrinkedArmature * pTargetBlock);

		void SetFrom(const ShrinkedArmature * pSourceBlock, const ShrinkedArmature * pTargetBlock);

		const ShrinkedArmature& SourceParts() const { return *m_sParts; }
		const ShrinkedArmature& TargetParts() const { return *m_cParts; }
	};

	class BlockizedCcaArmatureTransform : public BlockizedArmatureTransform
	{
	public:
		std::vector<PcaCcaMap> Maps;

		std::unique_ptr<IArmaturePartFeature> pInputExtractor, pOutputExtractor;
	public:

		using BlockizedArmatureTransform::BlockizedArmatureTransform;

		virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame) override;

		virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, float frame_time);
	};

	namespace ArmaturePartFeatures
	{
		class PerceptiveVector : public IArmaturePartFeature
		{
		private:
			CharacterController			*m_controller;
		public:
			float						Segma;
			bool						Quadratic;
			bool						Velocity;
		public:

			typedef BoneFeatures::GblPosFeature InputFeatureType;
			typedef CharacterFeature CharacterFeatureType;

			PerceptiveVector(CharacterController& controller);

			virtual int GetDimension() const;

			virtual int GetDimension(_In_ const ArmaturePart& block) const override;

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override;

			// Inherited via IArmaturePartFeature
			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override;
		};
	}

	class CharacterClipinfo;

	// for >=0, to sepecific part
	enum PvInputTypeEnum
	{
		ActiveAndDrivenParts = -2,
		ActiveParts = -3,
		NoInputParts = -1,
	};

	//Part to Part transform
	struct P2PTransform
	{
		int DstIdx, SrcIdx;
		Eigen::MatrixXf HomoMatrix; // homogenians transfrom matrix
	};

	class CharacterController;

	class PartilizedTransformer;

	void BlendFrame(IArmaturePartFeature& feature, const ShrinkedArmature& parts, ArmatureFrameView target, array_view<double> t, ArmatureFrameConstView f0, ArmatureFrameConstView f1);

	struct CharacterActionTrackerParameters
	{
		int								SubStep;

		// Particales initialization parameters
		int								TimelineSubdiv;
		int								VtSubdiv;
		int								ScaleSubdiv;

		double							ScaleMean;
		double							ScaleStdev;
		double							VtMean;
		double							VtStdev;

		double							DefaultSquareError;
	};

	class PartilizedTransformer : public BlockizedArmatureTransform
	{
	public:
		using TrackerScalarType = IGestureTracker::ScalarType;
		using TrackerVectorType = Eigen::Matrix<TrackerScalarType, 1, -1>;

		~PartilizedTransformer();

		std::vector<P2PTransform> ActiveParts;	// Direct controlled by input armature, with stylized IK

		PartilizedTransformer(const ShrinkedArmature& sParts, CharacterController & controller);

		void SetupTrackers(double expectedError, int stepSubdiv, double vtStep, double scaleStep, double vtStDev, double scaleStDev, double tInitDistSubdiv, int vtInitDistSubdiv, int scaleInitDistSubdiv);

		virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame) override;

		virtual void Transform(_Out_ frame_view target_frame, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, float frame_time) override;

		void SetTrackerVisualization();

		void DrivePartsTrackers(TrackerVectorType &_x, float frame_time, frame_view target_frame);
		double DrivePartsTracker(int whichTracker, TrackerVectorType & _x, float frame_time, Causality::ArmatureFrameView target_frame);

		void DriveAccesseryPart(Causality::ArmaturePart & cpart, Eigen::RowVectorXd &Xd, frame_view target_frame);

		void DriveActivePartSIK(Causality::ArmaturePart & cpart, frame_view target_frame, Eigen::RowVectorXf &xf, bool computeVelocity = false);

		void SetHandleVisualization(Causality::ArmaturePart & cpart, Eigen::RowVectorXf &xf);

		static void TransformCtrlHandel(Eigen::RowVectorXf &xf, const Eigen::MatrixXf& homoMatrix);

		void GenerateDrivenAccesseryControl();

		void EnableTracker(int whichTracker);
		void EnableTracker(const std::string& animName);

		void ResetTrackers();

	public:
		std::vector<P2PTransform> DrivenParts;	// These parts will be drive by active parts on Cca base than stylized IK
		std::vector<P2PTransform> AccesseryParts; // These parts will be animated based on active parts (and driven parts?)

		typedef Eigen::RowVectorXf InputVectorType;
		InputVectorType GetInputVector(_In_ const P2PTransform& Ctrl, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, _In_ float frame_time, int has_velocity) const;

		InputVectorType GetCharacterInputVector(_In_ const P2PTransform& Ctrl, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, _In_ float frame_time, int has_velocity) const;

		double GetInputKinectEnergy(_In_ const P2PTransform& Ctrl, _In_ const_frame_view source_frame, _In_ const_frame_view last_frame, _In_ double frame_time) const;
	private:
		typedef std::pair<Vector3, Vector3> LineSegment;

		CharacterController			&m_controller;
		array_view<LineSegment>		m_handles;
		MatrixVisualizer			*m_matvis;

		
		double						m_updateFreq;
		float						m_updateFreqf;
		
		LowPassFilter<double>		m_speedFilter;

		// Output Character Joint Filters
		vector<LowPassFilter<Quaternion, float>>
									m_jointFilters;;


		
		ArmatureFrame			m_ikDrivedFrame;
		
		ArmatureFrame			m_trackerFrame;

		typedef std::shared_ptr<IArmaturePartFeature> FeaturePtr;

		FeaturePtr	m_pInputF;
		
		FeaturePtr	m_pActiveF;
		
		FeaturePtr	m_pDrivenF;
		
		FeaturePtr	m_pAccesseryF;
		
		FeaturePtr  m_pTrackerFeature;

		bool		m_useTracker;
		int			m_currentTracker;
		// Tracker for different animations 
		vector<CharacterActionTracker>
					m_Trackers;

		// Tracker Switcher
		double		m_trackerSwitchThreshold;		// Confident threshold
		double		m_trackerSwitchTimeThreshold;	// Time threshold

		double		m_lowConfidentTime;
		int			m_lowConfidentFrameCount;

		Eigen::MatrixXd	
					m_trackerConfidents;
	};
}