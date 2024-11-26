// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>

#include <memory>
#include <mutex>

namespace AsyncExHistos {

class HistoWriter {
  std::vector<std::shared_ptr<TH1>> fHistos;
  std::string fFilename = "AsyncExampleHistos.root";
  std::mutex fMutex;

  HistoWriter() { TH1::AddDirectory(false); }

public:
  static HistoWriter &GetInstance()
  {
    static HistoWriter writer;
    return writer;
  }

  void SetFilename(std::string filename) { fFilename = std::move(filename); }

  void RegisterHisto(std::shared_ptr<TH1> histo)
  {
    std::scoped_lock lock{fMutex};
    fHistos.push_back(std::move(histo));
  }

  void WriteHistos(std::string const &filename = "")
  {
    std::scoped_lock lock{fMutex};
    if (fHistos.empty()) return;

    std::sort(fHistos.begin(), fHistos.end(),
              [](std::shared_ptr<const TH1> const &a, std::shared_ptr<const TH1> const &b) -> bool {
                return std::strcmp(a->GetName(), b->GetName()) < 0;
              });

    TFile file(filename.empty() ? fFilename.c_str() : filename.c_str(), "RECREATE");
    std::shared_ptr<TH1> currentHisto = fHistos.front();
    for (const auto &histo : fHistos) {
      if (currentHisto == histo) {
        continue;
      } else if (strcmp(currentHisto->GetName(), histo->GetName()) == 0) {
        currentHisto->Add(histo.get());
      } else {
        file.WriteTObject(currentHisto.get());
        currentHisto = histo;
      }
    }
    file.WriteTObject(currentHisto.get());
    file.Write();

    fHistos.clear();
  }
};

template <typename Histo>
void registerHisto(std::shared_ptr<Histo> histo)
{
  HistoWriter::GetInstance().RegisterHisto(histo);
}

} // namespace AsyncExHistos
