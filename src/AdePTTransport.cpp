// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTTransport.h>
#include <AdePT/benchmarking/TestManager.h>
#include <AdePT/benchmarking/TestManagerStore.h>

#include <VecGeom/management/BVHManager.h>
#include "VecGeom/management/GeoManager.h"
#include <VecGeom/gdml/Frontend.h>

#include "AdePT/copcore/SystemOfUnits.h"

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>

AdePTTransport::AdePTTransport()
{
}

AdePTTransport::~AdePTTransport()
{
  delete fScoring;
}

void AdePTTransport::AddTrack(int pdg, double energy, double x, double y, double z, double dirx, double diry,
                                double dirz)
{
  fBuffer.toDevice.emplace_back(pdg, energy, x, y, z, dirx, diry, dirz);
  if (pdg == 11)
    fBuffer.nelectrons++;
  else if (pdg == -11)
    fBuffer.npositrons++;
  else if (pdg == 22)
    fBuffer.ngammas++;

  if (fBuffer.toDevice.size() >= fBufferThreshold) {
    if (fDebugLevel > 0)
      std::cout << "Reached the threshold of " << fBufferThreshold << " triggering the shower" << std::endl;
    this->Shower(fIntegrationLayer.GetEventID());
  }
}

void AdePTTransport::Initialize(bool common_data)
{
  if (fInit) return;
  if (fMaxBatch <= 0) throw std::runtime_error("AdePTTransport::Initialize: Maximum batch size not set.");

  fNumVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();

  if (fNumVolumes == 0) throw std::runtime_error("AdePTTransport::Initialize: Number of geometry volumes is zero.");

  if (common_data) {
    std::cout << "=== AdePTTransport: initializing geometry and physics\n";
    // Initialize geometry on device
    if (!vecgeom::GeoManager::Instance().IsClosed())
      throw std::runtime_error("AdePTTransport::Initialize: VecGeom geometry not closed.");

    const vecgeom::cxx::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
    if (!InitializeGeometry(world))
      throw std::runtime_error("AdePTTransport::Initialize: Cannot initialize geometry on GPU");

    // Initialize G4HepEm
    if (!InitializePhysics()) throw std::runtime_error("AdePTTransport::Initialize cannot initialize physics on GPU");

    // Do the material-cut couple index mapping once
    // as well as set flags for sensitive volumes and region
    // Also set the mappings from sensitive volumes to hits and VecGeom to G4 indices
    int *sensitive_volumes = nullptr;

    // Check VecGeom geometry matches Geant4. Initialize auxiliary per-LV data. Initialize scoring map.  
    fIntegrationLayer.CheckGeometry(fg4hepem_state);
    VolAuxData *auxData = new VolAuxData[vecgeom::GeoManager::Instance().GetRegisteredVolumesCount()];
    fIntegrationLayer.InitVolAuxData(auxData, fg4hepem_state, fTrackInAllRegions, fGPURegionNames);

    // Initialize volume auxiliary data on device
    VolAuxArray::GetInstance().fNumVolumes = fNumVolumes;
    VolAuxArray::GetInstance().fAuxData    = auxData;
    VolAuxArray::GetInstance().InitializeOnGPU();

    // Print some settings
    std::cout << "=== AdePTTransport: buffering " << fBufferThreshold << " particles for transport on the GPU" << std::endl;
    std::cout << "=== AdePTTransport: maximum number of GPU track slots per thread: " << kCapacity << std::endl;
    return;
  }

  fIntegrationLayer.InitScoringData(VolAuxArray::GetInstance().fAuxData);

  std::cout << "=== AdePTTransport: initializing transport engine for thread: " << fIntegrationLayer.GetThreadID()
         << std::endl;

  // Initialize user scoring data
  fScoring     = new AdeptScoring(kHitBufferCapacity);
  fScoring_dev = fScoring->InitializeOnGPU();

  // Initialize the transport engine for the current thread
  InitializeGPU();

  fInit = true;
}

void AdePTTransport::InitBVH()
{
  vecgeom::cxx::BVHManager::Init();
  vecgeom::cxx::BVHManager::DeviceInit();
}

void AdePTTransport::Cleanup()
{
  if (!fInit) return;
  AdePTTransport::FreeGPU();
  fScoring->FreeGPU(fScoring_dev);
  delete[] fBuffer.fromDeviceBuff;
}

void AdePTTransport::Shower(int event)
{
  int tid = fIntegrationLayer.GetThreadID();
  if (fDebugLevel > 0 && fBuffer.toDevice.size() == 0) {
    std::cout << "[" << tid << "] AdePTTransport::Shower: No more particles in buffer. Exiting.\n";
    return;
  }

  if (event != fBuffer.eventId) {
    fBuffer.eventId    = event;
    fBuffer.startTrack = 0;
  } else {
    fBuffer.startTrack += fBuffer.toDevice.size();
  }

  int itr   = 0;
  int nelec = 0, nposi = 0, ngamma = 0;
  if (fDebugLevel > 0) {
    std::cout << "[" << tid << "] toDevice: " << fBuffer.nelectrons << " elec, " << fBuffer.npositrons << " posi, "
           << fBuffer.ngammas << " gamma\n";
  }
  if (fDebugLevel > 1) {
    for (auto &track : fBuffer.toDevice) {
      std::cout << "[" << tid << "] toDevice[ " << itr++ << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << std::endl;
    }
  }

  AdePTTransport::ShowerGPU(event, fBuffer);

  for (auto const &track : fBuffer.fromDevice) {
    if (track.pdg == 11)
      nelec++;
    else if (track.pdg == -11)
      nposi++;
    else if (track.pdg == 22)
      ngamma++;
  }
  if (fDebugLevel > 0) {
    std::cout << "[" << tid << "] fromDevice: " << nelec << " elec, " << nposi << " posi, " << ngamma << " gamma\n";
  }

  fIntegrationLayer.ReturnTracks(&(fBuffer.fromDevice), fDebugLevel);

  fBuffer.Clear();

  // fScoring->ClearGPU();
}
