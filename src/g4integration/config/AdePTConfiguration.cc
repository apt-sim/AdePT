// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/integration/AdePTConfigurationMessenger.hh>

#include <memory>

AdePTConfiguration::AdePTConfiguration()
    : fAdePTConfigurationMessenger(std::make_unique<AdePTConfigurationMessenger>(this))
{
}

AdePTConfiguration::~AdePTConfiguration() = default;
