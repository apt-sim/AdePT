// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_HEPEM_HOST_DATA_HH
#define ADEPT_HEPEM_HOST_DATA_HH

#include <memory>

struct G4HepEmConfig;
struct G4HepEmData;
struct G4HepEmParameters;

namespace AsyncAdePT {

/// @brief Owns the host-side HepEm tables rebuilt from a `G4HepEmConfig`.
/// @details
/// The Geant4 integration side prepares one of these objects before the shared
/// transport is created. The helper owns the rebuilt `G4HepEmData`, while the
/// `G4HepEmParameters` pointer is only borrowed from the `G4HepEmConfig`.
///
/// AdePT also creates the GPU mirror for the borrowed parameters. That GPU-side
/// allocation is released here in the destructor, because this helper owns the
/// corresponding upload lifecycle even though the host parameters themselves are
/// still owned by G4HepEm.
///
/// Cleanup is intentionally split:
/// - `DataDeleter` performs the deep cleanup of the owned `G4HepEmData`
///   and then deletes the outer `G4HepEmData` allocation.
/// - `~HepEmHostData()` releases only the GPU mirror of the borrowed
///   `G4HepEmParameters`.
class HepEmHostData {
public:
  /// @brief Rebuild all host-side HepEm tables needed by AdePT from the given config.
  explicit HepEmHostData(G4HepEmConfig *hepEmConfig);

  /// @brief Release the GPU mirror of the borrowed HepEm parameters.
  /// @details
  /// This performs a CUDA-side free through `FreeG4HepEmParametersOnGPU`.
  /// It does not free the host-side `G4HepEmParameters`, because those remain
  /// owned by the original `G4HepEmConfig`.
  ~HepEmHostData();

  HepEmHostData(const HepEmHostData &)            = delete;
  HepEmHostData &operator=(const HepEmHostData &) = delete;

  HepEmHostData(HepEmHostData &&) noexcept            = default;
  HepEmHostData &operator=(HepEmHostData &&) noexcept = default;

  /// @brief Access the owned host-side HepEm data tables.
  G4HepEmData *GetData() const { return fData.get(); }

  /// @brief Access the borrowed HepEm parameter block from the Geant4 config.
  G4HepEmParameters *GetParameters() const { return fParameters; }

private:
  /// @brief Deletes the outer `G4HepEmData` object after first freeing all tables it owns.
  /// @details
  /// `FreeG4HepEmData` releases both the host-side tables and any device-side
  /// mirrors embedded in the `G4HepEmData` object, but it does not delete the outer
  /// `G4HepEmData` allocation itself. This deleter performs both steps for the
  /// owned `fData` member. It does not touch the separately borrowed parameter block.
  struct DataDeleter {
    void operator()(G4HepEmData *data) const;
  };

  /// Owned host-side HepEm tables rebuilt for AdePT.
  std::unique_ptr<G4HepEmData, DataDeleter> fData;

  /// Non-owning pointer to the G4HepEmParameters that are owned by G4HepEm itself, not AdePT.
  G4HepEmParameters *fParameters{nullptr};
};

} // namespace AsyncAdePT

#endif
