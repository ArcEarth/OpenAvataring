#pragma once
#include <Eigen\Core>
#include "Causality\Animations.h"
#include "ArmatureParts.h"

namespace Causality
{
	class IGestureTracker
	{
	public:
	public:
		typedef double ScalarType;
		typedef double LikilihoodScalarType;
		typedef Eigen::Matrix<LikilihoodScalarType, -1, 1> LikihoodsType;
		typedef Eigen::Matrix<ScalarType, -1, -1> MatrixType;
		typedef Eigen::Matrix<ScalarType, 1, -1, Eigen::AutoAlign | Eigen::RowMajor> TrackingVectorType;
		typedef Eigen::Block<MatrixType, 1, -1> TrackingVectorBlockType;

		typedef Eigen::Matrix<ScalarType, 1, -1> InputVectorType;

		virtual ~IGestureTracker();

	public:
		// aka, Initialize, discard all history information, just initialize with input, reset all state
		virtual void Reset(const InputVectorType& input) = 0;
		// Step forward the tracking state from t to t+1
		// return the confident of current tracking
		virtual ScalarType Step(const InputVectorType& input, ScalarType dt) = 0;
		// Get the tracking state
		virtual const TrackingVectorType& CurrentState() const = 0;
	};

	// Provide base methods for Particle Filter
	// 
	// The sample matrix S is N x (Dim+1)
	// Where N is number of Particals
	// Dim is the state vector dimension
	// S.col(0), the first column of S stores the weights
	// S.row(i) is a particale
	// S(i, 0) is the particale weight
	// S(i, 1...Dim) is the state vector
	class ParticaleFilterBase : public IGestureTracker
	{
	public:
		~ParticaleFilterBase();

		ScalarType Step(const InputVectorType& input, ScalarType dt) override;

		// Weighted Avaerage State
		const TrackingVectorType& CurrentState() const override;
		const TrackingVectorType& MostLikilyState() const;

		int*  GetTopKStates(int k) const;

		//virtual void Reset(const InputVectorType& input) = 0;

		const MatrixType& GetSampleMatrix() const;
	
	protected: // Interfaces
		virtual void SetInputState(const InputVectorType& input, ScalarType dt) = 0;
		// Get the likilihood of partical state x in current time with pre-seted input state
		virtual ScalarType Likilihood(int idx, const TrackingVectorBlockType &x) = 0;

		virtual void Progate(TrackingVectorBlockType& x) = 0;

	public:
		MatrixType m_liks;

	protected:
		ScalarType StepParticals();

		double ExtractMLE();

		void Resample(_Out_ LikihoodsType& resampledLik, _Out_ MatrixType& resampled, _In_ const LikihoodsType& sampleLik, _In_ const MatrixType& sample);

		int			m_maxK;
		mutable vector<int> m_srtIdxes;
		LikihoodsType m_sampleLiks;
		LikihoodsType m_newSampleLiks;
		MatrixType  m_sample;
		MatrixType  m_newSample;
		// Mean state 
		TrackingVectorType m_waState;
		TrackingVectorType m_mleState;
	};
}