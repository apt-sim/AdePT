// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/AdePTConfiguration.hh>
#include <AdePT/g4integration/config/AdePTConfigurationMessenger.hh>

#include <memory>

AdePTConfiguration::AdePTConfiguration()
    : fAdePTConfigurationMessenger(std::make_unique<AdePTConfigurationMessenger>(this))
{
}

AdePTConfiguration::~AdePTConfiguration() = default;
