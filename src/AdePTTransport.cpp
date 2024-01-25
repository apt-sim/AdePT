// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTTransport.h>

#include <VecGeom/management/BVHManager.h>
#include "VecGeom/management/GeoManager.h"
#include <VecGeom/gdml/Frontend.h>

#include "CopCore/SystemOfUnits.h"

#include <G4Threading.hh>
#include <G4EventManager.hh>
#include <G4Event.hh>

#include <G4HepEmData.hh>
#include <G4HepEmState.hh>
#include <G4HepEmStateInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmMatCutData.hh>

#include <AdePT/base/TestManager.h>
#include <AdePT/base/TestManagerStore.h>

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
      G4cout << "Reached the threshold of " << fBufferThreshold << " triggering the shower" << G4endl;
    this->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
  }
}

void AdePTTransport::Initialize(bool common_data)
{
  if (fInit) return;
  if (fMaxBatch <= 0) throw std::runtime_error("AdePTTransport::Initialize: Maximum batch size not set.");

  fNumVolumes = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();

  if (fNumVolumes == 0) throw std::runtime_error("AdePTTransport::Initialize: Number of geometry volumes is zero.");

  if (common_data) {
    G4cout << "=== AdePTTransport: initializing geometry and physics\n";
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
    fIntegrationLayer.InitVolAuxData(auxData, fg4hepem_state);

    // VolAuxData *auxData = new VolAuxData[vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();];
    // AdePTGeant4Integration::InitVolAuxData(fg4hepem_state, auxData);

    // Initialize volume auxiliary data on device
    VolAuxArray::GetInstance().fNumVolumes = fNumVolumes;
    VolAuxArray::GetInstance().fAuxData    = auxData;
    VolAuxArray::GetInstance().InitializeOnGPU();

    // Print some settings
    G4cout << "=== AdePTTransport: buffering " << fBufferThreshold << " particles for transport on the GPU" << G4endl;
    G4cout << "=== AdePTTransport: maximum number of GPU track slots per thread: " << kCapacity << G4endl;
    return;
  }

  fIntegrationLayer.InitScoringData(VolAuxArray::GetInstance().fAuxData);
  //fIntegrationLayer.Initialize(fg4hepem_state);

  G4cout << "=== AdePTTransport: initializing transport engine for thread: " << G4Threading::G4GetThreadId()
         << G4endl;

  // Initialize user scoring data
  // fScoring     = new AdeptScoring(fGlobalNumSensitive, &fglobal_volume_to_hit_map,
  // vecgeom::GeoManager::Instance().GetPlacedVolumesCount());
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
  constexpr double tolerance = 10. * vecgeom::kTolerance;

  int tid = G4Threading::G4GetThreadId();
  if (fDebugLevel > 0 && fBuffer.toDevice.size() == 0) {
    G4cout << "[" << tid << "] AdePTTransport::Shower: No more particles in buffer. Exiting.\n";
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
    G4cout << "[" << tid << "] toDevice: " << fBuffer.nelectrons << " elec, " << fBuffer.npositrons << " posi, "
           << fBuffer.ngammas << " gamma\n";
  }
  if (fDebugLevel > 1) {
    for (auto &track : fBuffer.toDevice) {
      G4cout << "[" << tid << "] toDevice[ " << itr++ << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << G4endl;
    }
  }

  AdePTTransport::ShowerGPU(event, fBuffer);

  // if (fDebugLevel > 1) fScoring->Print();
  for (auto const &track : fBuffer.fromDevice) {
    if (track.pdg == 11)
      nelec++;
    else if (track.pdg == -11)
      nposi++;
    else if (track.pdg == 22)
      ngamma++;
  }
  if (fDebugLevel > 0) {
    G4cout << "[" << tid << "] fromDevice: " << nelec << " elec, " << nposi << " posi, " << ngamma << " gamma\n";
  }

  // Build the secondaries and put them back on the Geant4 stack
  int i = 0;
  for (auto const &track : fBuffer.fromDevice) {
    if (fDebugLevel > 1) {
      G4cout << "[" << tid << "] fromDevice[ " << i++ << "]: pdg " << track.pdg << " energy " << track.energy
             << " position " << track.position[0] << " " << track.position[1] << " " << track.position[2]
             << " direction " << track.direction[0] << " " << track.direction[1] << " " << track.direction[2] << G4endl;
    }
    G4ParticleMomentum direction(track.direction[0], track.direction[1], track.direction[2]);

    G4DynamicParticle *dynamique =
        new G4DynamicParticle(G4ParticleTable::GetParticleTable()->FindParticle(track.pdg), direction, track.energy);

    G4ThreeVector posi(track.position[0], track.position[1], track.position[2]);
    // The returned track will be located by Geant4. For now we need to
    // push it to make sure it is not relocated again in the GPU region
    posi += tolerance * direction;

    G4Track *secondary = new G4Track(dynamique, 0, posi);
    secondary->SetParentID(-99);

    G4EventManager::GetEventManager()->GetStackManager()->PushOneTrack(secondary);
  }

  fBuffer.Clear();

  // fScoring->ClearGPU();
}
