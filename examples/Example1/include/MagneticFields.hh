// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//
#ifndef MAGNETICFIELDS_H
#define MAGNETICFIELDS_H

#include "G4MagneticField.hh"

#include "G4ThreeVector.hh"

#ifdef ADEPT_USE_EXT_BFIELD

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

#include <fstream>

using cpu_field_t = covfie::field<covfie::backend::affine<covfie::backend::linear<
        covfie::backend::strided<covfie::vector::size3, covfie::backend::array<covfie::vector::float3>>>>>;

#endif 

class G4GenericMessenger;


class IField {
public:
  virtual ~IField() = default;
  virtual void GetFieldValue(const G4double point[4], double* bField) const = 0;
};

class UniformField : public IField {
public:
  UniformField(const G4ThreeVector& field) : fField(field) {}

  void GetFieldValue(const G4double[4], double* bField) const override {
        bField[0] = fField.x();
        bField[1] = fField.y();
        bField[2] = fField.z();
        printf("Field value %f %f %f\n", bField[0], bField[1], bField[2]);
    }

private:
  G4ThreeVector fField;
};

class CovfieField : public IField {
public:
  CovfieField(const std::string& filename) {
    if (filename != "") {
#ifdef ADEPT_USE_EXT_BFIELD
    // Initialize field map using Covfie library
    std::ifstream ifs(filename, std::ifstream::binary);
    if (!ifs.good()) throw std::runtime_error("Failed to open field file: " + filename);

    cpuField = cpu_field_t(ifs);
    fFieldView = std::make_unique<cpu_field_t::view_t>(cpuField);
#endif
    }
  }

  void GetFieldValue(const G4double point[4], double* bField) const override {
#ifdef ADEPT_USE_EXT_BFIELD
    auto field_value = fFieldView->at(point[0], point[1], point[2]);
    bField[0] = field_value[0];
    bField[1] = field_value[1];
    bField[2] = field_value[2];
#endif
  }

private:
#ifdef ADEPT_USE_EXT_BFIELD
    cpu_field_t cpuField;
    std::unique_ptr<cpu_field_t::view_t> fFieldView;
#endif
};

class MagneticField : public G4MagneticField {
public:
    MagneticField(std::unique_ptr<IField> field) : fField(std::move(field)) {}
    ~MagneticField() override = default;

    void GetFieldValue(const G4double point[4], double* bField) const override {
        fField->GetFieldValue(point, bField);
    }

private:
    std::unique_ptr<IField> fField;
};

#endif // MAGNETICFIELDS_H