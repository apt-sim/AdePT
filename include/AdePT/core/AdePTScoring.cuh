// Templates for the AdePTScoring CUDA methods

namespace adept_scoring
{
  template <typename Scoring>
  Scoring *InitializeOnGPU(Scoring &scoring){}

  template <typename Scoring>
  void FreeGPU(Scoring *scoring_dev){}

  template <typename Scoring>
  __device__ void RecordHit(Scoring *scoring_dev, char aParticleType, double aStepLength, double aTotalEnergyDeposit,
                          vecgeom::NavigationState const *aPreState, vecgeom::Vector3D<Precision> *aPrePosition,
                          vecgeom::Vector3D<Precision> *aPreMomentumDirection,
                          vecgeom::Vector3D<Precision> *aPrePolarization, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const *aPostState, vecgeom::Vector3D<Precision> *aPostPosition,
                          vecgeom::Vector3D<Precision> *aPostMomentumDirection,
                          vecgeom::Vector3D<Precision> *aPostPolarization, double aPostEKin, double aPostCharge){}

  template <typename Scoring>
  __device__ void EndOfIterationGPU(Scoring *scoring_dev);

  template <typename Scoring, typename IntegrationLayer>
  void EndOfIteration(Scoring &scoring, Scoring *scoring_dev, cudaStream_t stream, IntegrationLayer integration);

  template <typename Scoring, typename IntegrationLayer>
  void EndOfTransport(Scoring &scoring, Scoring *scoring_dev, cudaStream_t stream, IntegrationLayer integration);
}
