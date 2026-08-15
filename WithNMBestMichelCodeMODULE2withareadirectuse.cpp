//stores output in a root file
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TPaveStats.h>
#include <TGaxis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <memory>
#include <cmath>

using std::cout;
using std::cerr;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0,10,7,2,6,3,8,9,11,4,5,1};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int BS_UNCERTAINTY = 5;       // Baseline uncertainty (ADC)
const int EV61_THRESHOLD = 1200;    // Beam on if channel 22 > this (ADC)
const double MUON_ENERGY_THRESHOLD = 50; // Min PMT energy for muon (p.e.)
const double MICHEL_ENERGY_MIN = 50;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX_DT = 500; // Max PMT energy for dt plots (p.e.)
const double MICHEL_DT_MIN = 2.0;       // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform

// Good event cut constants
const double PEAK_POSITION_RMS_CUT = 2.5;
const double AREA_HEIGHT_RATIO_CUT = 1.2;

// Veto panel thresholds
const std::vector<double> TOP_VP_THRESHOLDS = {1000, 1000};       // Channels 12-13 (ADC)
const std::vector<double> WIDE_SIDE_VP_THRESHOLDS = {1100, 1500, 1000, 1100}; // Channels 14-17 (ADC)
const std::vector<double> THIN_SIDE_VP_THRESHOLDS = {1000, 750, 750, 750};    // Channels 18-21 (ADC)
const double FIT_MIN = 2.0; // Fit range min (µs)
const double FIT_MAX = 16.0; // Fit range max (µs)

// Generate unique output directory with timestamp
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

// SPE fitting functions
Double_t fitGauss(Double_t *x, Double_t *par) {
    return par[0] * TMath::Gaus(x[0], par[1], par[2]);
}

Double_t six_fit_func(Double_t *x, Double_t *par) {
    return (par[0] * TMath::Gaus(x[0], par[1], par[2]) +
           par[3] * TMath::Gaus(x[0], par[4], par[5]));
}

Double_t eight_fit_func(Double_t *x, Double_t *par) {
    return (par[0] * TMath::Gaus(x[0], par[1], par[2]) +
           par[3] * TMath::Gaus(x[0], par[4], par[5]) +
           par[6] * TMath::Gaus(x[0], 2.0 * par[4], TMath::Sqrt(2.0 * par[5]*par[5] - par[2]*par[2])) +
           par[7] * TMath::Gaus(x[0], 3.0 * par[4], TMath::Sqrt(3.0 * par[5]*par[5] - 2.0 * par[2]*par[2])));
}

// Exponential fit function: N0 * exp(-t/tau) + C (t, tau in µs)
Double_t ExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0] / par[1]) + par[2];
}

// Pulse structure
struct pulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    double trigger;        // Trigger type
    double top_vp_energy;  // Top veto energy (ADC)
    double wide_side_vp_energy; // Wide side veto energy (ADC)
    double thin_side_vp_energy; // Thin side veto energy (ADC)
    double all_vp_energy;  // All veto energy (ADC)
    double last_muon_time; // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
    bool veto_hit[10];     // Which veto panels were hit (channels 12-21)
    bool is_good_event;    // Good event flag
};

// Utility functions
template<typename T>
double getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, int> count;
    for (const auto& val : v) count[val]++;
    T most_common = v[0];
    int max_count = 0;
    for (const auto& pair : count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    return max_count > 1 ? most_common : getAverage(v);
}

template<typename T>
double variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    double mean = getAverage(v);
    double sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

void CalculateMeanAndRMS(const vector<Double_t> &data, Double_t &mean, Double_t &rms) {
    if (data.empty()) {
        mean = 0;
        rms = 0;
        return;
    }
    mean = 0.0;
    for (const auto &value : data) mean += value;
    mean /= data.size();
    
    rms = 0.0;
    for (const auto &value : data) rms += pow(value - mean, 2);
    rms = sqrt(rms / data.size());
}

// Good event quality cut function using direct ROOT branches
bool isGoodEvent(const Double_t* pulseH, const Double_t* area, const Int_t* peakPosition, 
                 const Double_t* mu1, const Double_t* baselineRMS) {
    
    // Collect PMT pulse information
    vector<Double_t> pmt_peaks(N_PMTS, 0);
    vector<Double_t> pmt_energies(N_PMTS, 0);
    vector<Int_t> pmt_peak_positions(N_PMTS, -1);
    
    for (int pmt = 0; pmt < N_PMTS; pmt++) {
        int channel = PMT_CHANNEL_MAP[pmt];
        if (mu1[pmt] > 0) {
            pmt_peaks[pmt] = pulseH[channel] / mu1[pmt];
            pmt_energies[pmt] = area[channel] / mu1[pmt];
            pmt_peak_positions[pmt] = peakPosition[channel];
        }
    }
    
    // Condition A: At least 3 PMTs with peak > 2 p.e.
    int countAbove2PE = 0;
    for (int pmt = 0; pmt < N_PMTS; pmt++) {
        if (pmt_peaks[pmt] > 0 && pmt_peaks[pmt] > 2 * mu1[pmt]) {
            countAbove2PE++;
        }
    }

    if (countAbove2PE >= 3) {
        // Check peak position consistency
        vector<Double_t> peakPositions;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            if (pmt_peaks[pmt] > 0) {
                peakPositions.push_back(pmt_peak_positions[pmt]);
            }
        }
        
        if (!peakPositions.empty()) {
            Double_t dummyMean;
            Double_t current_rms;
            CalculateMeanAndRMS(peakPositions, dummyMean, current_rms);
            if (current_rms < PEAK_POSITION_RMS_CUT) return true;
        }
    } 
    else {
        // Condition B: At least 3 PMTs with peak > 3*RMS and area/height ratio > 1.2
        int countConditionB = 0;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            int channel = PMT_CHANNEL_MAP[pmt];
            if (pmt_peaks[pmt] > 0) {
                double peak_adc = pulseH[channel];
                double area_adc = area[channel];
                if (peak_adc > 3 * baselineRMS[channel] && 
                    (area_adc / peak_adc) > AREA_HEIGHT_RATIO_CUT) {
                    countConditionB++;
                }
            }
        }

        if (countConditionB >= 3) {
            // Check peak position consistency
            vector<Double_t> peakPositions;
            for (int pmt = 0; pmt < N_PMTS; pmt++) {
                if (pmt_peaks[pmt] > 0) {
                    peakPositions.push_back(pmt_peak_positions[pmt]);
                }
            }
            
            if (!peakPositions.empty()) {
                Double_t dummyMean;
                Double_t current_rms;
                CalculateMeanAndRMS(peakPositions, dummyMean, current_rms);
                if (current_rms < PEAK_POSITION_RMS_CUT) return true;
            }
        }
    }

    return false;
}

// Create output directory
void createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st) != 0) {
        if (mkdir(dirName.c_str(), 0755) != 0) {
            cerr << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// SPE calibration function
bool performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        return false;
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        delete calibFile;
        return false;
    }

    string speDir = OUTPUT_DIR + "/SPE_Fits";
    gSystem->mkdir(speDir.c_str(), kTRUE);

    TH1F *histArea[N_PMTS];
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                             Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
        histArea[i]->SetLineColor(kRed);
    }

    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
        }
    }

    Int_t defaultErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kError;

    string individualPlotsDir = speDir + "/Individual";
    gSystem->mkdir(individualPlotsDir.c_str(), kTRUE);

    TCanvas *c_combined = new TCanvas("c_combined", "SPE Fits - Combined", 1200, 800);
    c_combined->Divide(4, 3);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    // Store individual canvases for ROOT file
    vector<TCanvas*> indivCanvases;

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        c_combined->cd(i+1);

        TF1 *f1 = new TF1("f1", fitGauss, -50, 50, 3);
        f1->SetParameters(1500, 0, 25);
        f1->SetParNames("A0", "#mu_{0}", "#sigma_{0}");
        histArea[i]->Fit(f1, "Q", "", -50, 50);

        TF1 *f6 = new TF1("f6", six_fit_func, -50, 200, 6);
        f6->SetParameters(f1->GetParameter(0), f1->GetParameter(1), f1->GetParameter(2),
                        1800, 70, 30);
        f6->SetParNames("A0", "#mu_{0}", "#sigma_{0}", "A1", "#mu_{1}", "#sigma_{1}");
        histArea[i]->Fit(f6, "Q", "", -50, 200);

        TF1 *f8 = new TF1("f8", eight_fit_func, -50, 400, 8);
        f8->SetParameters(f6->GetParameter(0), f6->GetParameter(1), f6->GetParameter(2),
                        f6->GetParameter(3), f6->GetParameter(4), f6->GetParameter(5),
                        200, 50);
        f8->SetParNames("A0", "#mu_{0}", "#sigma_{0}", "A1", "#mu_{1}", "#sigma_{1}", "A2", "A3");
        f8->SetLineColor(kBlue);
        histArea[i]->Fit(f8, "Q", "", -50, 400);

        mu1[i] = f8->GetParameter(4);
        mu1_err[i] = f8->GetParError(4);

        histArea[i]->Draw();
        f8->Draw("same");

        TLatex tex;
        tex.SetTextFont(42);
        tex.SetTextSize(0.04);
        tex.SetNDC();
        tex.DrawLatex(0.15, 0.85, Form("PMT %d", i+1));
        tex.DrawLatex(0.15, 0.80, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]));

        // Create individual canvas for this PMT
        TCanvas *c_indiv = new TCanvas(Form("c_pmt%d", i+1), Form("PMT %d SPE Fit", i+1), 1200, 800);
        histArea[i]->Draw();
        f8->Draw("same");
        tex.DrawLatex(0.15, 0.85, Form("PMT %d", i+1));
        tex.DrawLatex(0.15, 0.80, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]));

        // Save individual plot as PNG
        string indivPlotName = individualPlotsDir + Form("/PMT%d_SPE_Fit.png", i+1);
        c_indiv->SaveAs(indivPlotName.c_str());
        cout << "Saved individual SPE plot: " << indivPlotName << endl;

        // Store canvas for ROOT file
        indivCanvases.push_back(c_indiv);
        delete f1;
        delete f6;
        delete f8;
    }

    string combinedPlotName = speDir + "/SPE_Fits_Combined.png";
    c_combined->SaveAs(combinedPlotName.c_str());
    cout << "Saved combined SPE plot: " << combinedPlotName << endl;

    gErrorIgnoreLevel = defaultErrorLevel;

    // Save calibration results to ROOT file
    TFile *calibRootFile = new TFile((OUTPUT_DIR + "/calibration_histograms.root").c_str(), "RECREATE");
    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]) {
            histArea[i]->Write();
        }
    }
    // Save the combined canvas
    c_combined->Write("SPE_Fits_Combined");
    // Save individual canvases
    for (size_t i = 0; i < indivCanvases.size(); i++) {
        if (indivCanvases[i]) {
            indivCanvases[i]->Write(Form("PMT%d_SPE_Fit", i+1));
        }
    }
    calibRootFile->Close();
    delete calibRootFile;
    cout << "Calibration histograms and canvases saved to: " << OUTPUT_DIR << "/calibration_histograms.root" << endl;

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]) delete histArea[i];
    }
    for (size_t i = 0; i < indivCanvases.size(); i++) {
        delete indivCanvases[i];
    }
    delete c_combined;
    calibFile->Close();
    delete calibFile;
    
    return true;
}

void createVetoPanelPlots(TH1D* h_veto_panel[10], const string& outputDir, TFile* outFile) {
    vector<TCanvas*> vetoCanvases;
    
    for (int i = 0; i < 10; i++) {
        TCanvas *c = new TCanvas(Form("c_veto_%d", i+12), Form("Veto Panel %d", i+12), 1200, 800);
        gStyle->SetOptStat(1111);
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->Draw("hist");
        string plotName = outputDir + Form("/Veto_Panel_%d.png", i+12);
        c->SaveAs(plotName.c_str());
        cout << "Saved veto panel plot: " << plotName << endl;
        vetoCanvases.push_back(c);
    }

    TCanvas *c_combined = new TCanvas("c_veto_combined", "Combined Veto Panel Energies", 1600, 1200);
    c_combined->Divide(4, 3);
    for (int i = 0; i < 10; i++) {
        c_combined->cd(i+1);
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->SetTitle("");
        h_veto_panel[i]->Draw("hist");
    }
    string combinedPlotName = outputDir + "/Combined_Veto_Panels.png";
    c_combined->SaveAs(combinedPlotName.c_str());
    cout << "Saved combined veto panel plot: " << combinedPlotName << endl;
    
    // Save veto canvases to ROOT file
    if (outFile) {
        outFile->cd();
        for (size_t i = 0; i < vetoCanvases.size(); i++) {
            if (vetoCanvases[i]) {
                vetoCanvases[i]->Write(Form("Veto_Panel_%d", i+12));
            }
        }
        c_combined->Write("Combined_Veto_Panels");
    }
    
    for (size_t i = 0; i < vetoCanvases.size(); i++) {
        delete vetoCanvases[i];
    }
    delete c_combined;
}

bool checkVetoHit(int channel, double energy) {
    if (channel >= 12 && channel <= 13) {
        return energy > TOP_VP_THRESHOLDS[channel-12];
    }
    else if (channel >= 14 && channel <= 17) {
        return energy > WIDE_SIDE_VP_THRESHOLDS[channel-14];
    }
    else if (channel >= 18 && channel <= 21) {
        return energy > THIN_SIDE_VP_THRESHOLDS[channel-18];
    }
    return false;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    createOutputDirectory(OUTPUT_DIR);

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file not found" << endl;
        return -1;
    }

    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    if (!performCalibration(calibFileName, mu1, mu1_err)) {
        cerr << "SPE calibration failed!" << endl;
        return -1;
    }

    cout << "\nSPE Calibration Results:\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }
    cout << endl;

    int num_muons = 0, num_michels = 0, num_events = 0;
    int num_good_events = 0;
    map<int, int> trigger_counts;

    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution;Energy (p.e.);Counts/100 p.e.", 550, -500, 5000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/8 p.e.", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaT", "Muon-Michel Time Difference;Time to Previous Muon (#mus);Counts/0.08 #mus", 200, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 16, 200, 0, 1000);
    TH1D* h_top_vp_muon = new TH1D("top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_wide_side_vp_muon = new TH1D("wide_side_vp_muon", "Wide Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_thin_side_vp_muon = new TH1D("thin_side_vp_muon", "Thin Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);
    TH1D* h_good_vs_bad = new TH1D("good_vs_bad", "Event Quality;Quality;Counts", 2, 0, 2);
    TH1D* h_peak_position_rms = new TH1D("peak_position_rms", "Peak Position RMS Distribution;RMS (samples);Counts", 100, 0, 10);
    
    TH1D* h_veto_panel[10];
    const char* veto_names[10] = {
        "Top Veto Panel 12", "Top Veto Panel 13",
        "Wide Side Veto Panel 14", "Wide Side Veto Panel 15", "Wide Side Veto Panel 16", "Wide Side Veto Panel 17",
        "Thin Side Veto Panel 18", "Thin Side Veto Panel 19", "Thin Side Veto Panel 20", "Thin Side Veto Panel 21"
    };
    
    for (int i = 0; i < 10; i++) {
        h_veto_panel[i] = new TH1D(Form("h_veto_panel_%d", i+12), 
                                  Form("%s;Energy (ADC);Counts", veto_names[i]), 
                                  200, 0, 8000);
    }

    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = TFile::Open(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file" << endl;
            f->Close();
            continue;
        }

        Int_t eventID, nSamples[23], peakPosition[23], triggerBits;
        Short_t adcVal[23][45];
        Double_t baselineMean[23], baselineRMS[23], pulseH[23], area[23];
        Long64_t nsTime;

        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);

        int numEntries = t->GetEntries();
        cout << "Processing " << numEntries << " entries..." << endl;
        
        double last_muon_time = 0.0;
        set<double> michel_muon_times;
        vector<pair<double, double>> muon_candidates;

        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;
            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;

            pulse p;
            p.start = nsTime / 1000.0;
            p.end = nsTime / 1000.0;
            p.energy = 0;
            p.number = 0;
            p.trigger = triggerBits;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;
            p.is_good_event = false;
            for (int i = 0; i < 10; i++) p.veto_hit[i] = false;

            vector<double> all_chan_energy;
            vector<double> top_vp_energy, wide_side_vp_energy, thin_side_vp_energy;
            vector<double> veto_energies(10, 0);

            // Process each channel - USING AREA BRANCH
            for (int iChan = 0; iChan < 23; iChan++) {
                
                // Check beam status (channel 22)
                if (iChan == 22) {
                    if (area[iChan] > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }
                
                // Process PMT channels (0-11) - USE AREA BRANCH
                if (iChan <= 11) {
                    // Convert ADC area to p.e. using calibration
                    double energy_pe = 0;
                    int pmt_idx = -1;
                    for (int k = 0; k < N_PMTS; k++) {
                        if (PMT_CHANNEL_MAP[k] == iChan) {
                            pmt_idx = k;
                            break;
                        }
                    }
                    
                    if (pmt_idx >= 0 && mu1[pmt_idx] > 0) {
                        energy_pe = area[iChan] / mu1[pmt_idx];
                    }
                    
                    if (energy_pe > 1) {
                        all_chan_energy.push_back(energy_pe);
                        p.number += 1;
                        p.energy += energy_pe;
                    }
                }
                
                // Process Top Veto Panels (12-13) - USE AREA BRANCH
                else if (iChan >= 12 && iChan <= 13) {
                    double veto_energy = area[iChan];
                    top_vp_energy.push_back(veto_energy);
                    veto_energies[iChan-12] = veto_energy;
                    p.veto_hit[iChan-12] = checkVetoHit(iChan, veto_energy);
                }
                
                // Process Wide Side Veto Panels (14-17) - USE AREA BRANCH
                else if (iChan >= 14 && iChan <= 17) {
                    double veto_energy = area[iChan];
                    wide_side_vp_energy.push_back(veto_energy);
                    veto_energies[iChan-12] = veto_energy;
                    p.veto_hit[iChan-12] = checkVetoHit(iChan, veto_energy);
                }
                
                // Process Thin Side Veto Panels (18-21) - USE AREA BRANCH
                else if (iChan >= 18 && iChan <= 21) {
                    double veto_energy = area[iChan];
                    thin_side_vp_energy.push_back(veto_energy);
                    veto_energies[iChan-12] = veto_energy;
                    p.veto_hit[iChan-12] = checkVetoHit(iChan, veto_energy);
                }
            }

            // Aggregate energies
            p.top_vp_energy = accumulate(top_vp_energy.begin(), top_vp_energy.end(), 0.0);
            p.wide_side_vp_energy = accumulate(wide_side_vp_energy.begin(), wide_side_vp_energy.end(), 0.0);
            p.thin_side_vp_energy = accumulate(thin_side_vp_energy.begin(), thin_side_vp_energy.end(), 0.0);
            p.all_vp_energy = p.top_vp_energy + p.wide_side_vp_energy + p.thin_side_vp_energy;

            // Apply good event cut using direct ROOT branches
            p.is_good_event = isGoodEvent(pulseH, area, peakPosition, mu1, baselineRMS);
            
            // Fill peak position RMS histogram for all events
            vector<Double_t> peakPositions;
            for (int pmt = 0; pmt < N_PMTS; pmt++) {
                int channel = PMT_CHANNEL_MAP[pmt];
                if (mu1[pmt] > 0 && pulseH[channel] > 0) {
                    peakPositions.push_back(peakPosition[channel]);
                }
            }
            if (!peakPositions.empty()) {
                Double_t dummyMean;
                Double_t current_rms;
                CalculateMeanAndRMS(peakPositions, dummyMean, current_rms);
                h_peak_position_rms->Fill(current_rms);
            }
            
            // Fill good vs bad histogram
            h_good_vs_bad->Fill(p.is_good_event ? 0 : 1);

            // Only process good events for physics analysis
            if (p.is_good_event) {
                num_good_events++;

                // Muon detection
                bool veto_hit = false;
                for (int i = 0; i < 10; i++) {
                    if (p.veto_hit[i]) {
                        veto_hit = true;
                        break;
                    }
                }

                if (p.energy > MUON_ENERGY_THRESHOLD && veto_hit) {
                    p.is_muon = true;
                    last_muon_time = p.start;
                    num_muons++;
                    muon_candidates.emplace_back(p.start, p.energy);
                    h_top_vp_muon->Fill(p.top_vp_energy);
                    h_wide_side_vp_muon->Fill(p.wide_side_vp_energy);
                    h_thin_side_vp_muon->Fill(p.thin_side_vp_energy);
                    for (int i = 0; i < 10; i++) {
                        if (p.veto_hit[i]) {
                            h_veto_panel[i]->Fill(veto_energies[i]);
                        }
                    }
                }

                // Michel electron detection
                double dt = p.start - last_muon_time;
                bool veto_low = true;
                for (int i = 0; i < 10; i++) {
                    if (p.veto_hit[i]) {
                        veto_low = false;
                        break;
                    }
                }

                bool is_michel_candidate = p.energy >= MICHEL_ENERGY_MIN &&
                                          p.energy <= MICHEL_ENERGY_MAX &&
                                          dt >= MICHEL_DT_MIN &&
                                          dt <= MICHEL_DT_MAX &&
                                          p.number >= 8 &&
                                          veto_low &&
                                          p.trigger != 1 &&
                                          p.trigger != 4 &&
                                          p.trigger != 8 &&
                                          p.trigger != 16;
                
                h_energy_vs_dt->Fill(dt, p.energy);
                bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

                if (is_michel_candidate) {
                    p.is_michel = true;
                    num_michels++;
                    michel_muon_times.insert(last_muon_time);
                    h_michel_energy->Fill(p.energy);
                }

                if (is_michel_for_dt) {
                    h_dt_michel->Fill(dt);
                }

                p.last_muon_time = last_muon_time;
            }
        }

        // Fill muon energy histogram for muons associated with Michel electrons
        for (const auto& muon : muon_candidates) {
            if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
                h_muon_energy->Fill(muon.second);
            }
        }

        cout << "File Statistics: Events=" << num_events << ", Good=" << num_good_events 
             << ", Muons=" << num_muons << ", Michels=" << num_michels << endl;
        f->Close();
        delete f;
    }

    cout << "\nTrigger Bits Distribution:\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }

    // Create all canvases and save both PNG and ROOT files
    TFile *outFile = new TFile((OUTPUT_DIR + "/all_histograms.root").c_str(), "RECREATE");
    
    // Store canvases for ROOT file
    vector<TCanvas*> canvases;

    // Muon Energy
    TCanvas *c_muon = new TCanvas("c_muon", "Muon Energy", 800, 600);
    gStyle->SetOptStat(1111);
    h_muon_energy->SetLineColor(kBlue);
    h_muon_energy->SetLineWidth(2);
    h_muon_energy->Draw();
    c_muon->SaveAs((OUTPUT_DIR + "/Muon_Energy.png").c_str());
    canvases.push_back(c_muon);

    // Michel Energy
    TCanvas *c_michel = new TCanvas("c_michel", "Michel Energy", 800, 600);
    gStyle->SetOptStat(1111);
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->SetLineWidth(2);
    h_michel_energy->Draw();
    c_michel->SaveAs((OUTPUT_DIR + "/Michel_Energy.png").c_str());
    canvases.push_back(c_michel);

    // Michel dt with exponential fit
    TCanvas *c_dt = new TCanvas("c_dt", "Michel dt", 800, 600);
    gStyle->SetOptStat(1111);
    gPad->SetLeftMargin(0.12);
    h_dt_michel->SetLineWidth(2);
    h_dt_michel->SetMarkerStyle(20);
    h_dt_michel->Draw("E1");
    c_dt->SetLogy();
    
    TF1 *expFit = nullptr;
    if (h_dt_michel->GetEntries() > 5) {
        double integral = h_dt_michel->Integral(h_dt_michel->FindBin(FIT_MIN), h_dt_michel->FindBin(FIT_MAX));
        double bin_width = h_dt_michel->GetBinWidth(1);
        double N0_init = integral * bin_width / (FIT_MAX - FIT_MIN);
        double C_init = 0;
        
        int bin_12 = h_dt_michel->FindBin(12.0);
        int bin_16 = h_dt_michel->FindBin(16.0);
        double min_content = 1e9;
        for (int i = bin_12; i <= bin_16; i++) {
            double content = h_dt_michel->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        expFit = new TF1("expFit", ExpFit, FIT_MIN, FIT_MAX, 3);
        expFit->SetParameters(N0_init, 2.2, C_init);
        expFit->SetParLimits(0, 0, N0_init * 100);
        expFit->SetParLimits(1, 0.1, 20.0);
        expFit->SetParLimits(2, 0, C_init * 10);
        expFit->SetLineColor(kRed);
        expFit->SetLineWidth(3);
        expFit->SetNpx(1000);

        h_dt_michel->Fit(expFit, "R0", "", FIT_MIN, FIT_MAX);
        expFit->Draw("SAME");
        
        cout << "\nExponential Fit Results:\n";
        cout << Form("τ = %.4f ± %.4f µs", expFit->GetParameter(1), expFit->GetParError(1)) << endl;
        cout << Form("χ²/NDF = %.2f", expFit->GetChisquare() / expFit->GetNDF()) << endl;
    }
    
    c_dt->SaveAs((OUTPUT_DIR + "/Michel_dt.png").c_str());
    canvases.push_back(c_dt);

    // Energy vs dt
    TCanvas *c_evsdt = new TCanvas("c_evsdt", "Energy vs dt", 800, 600);
    gStyle->SetOptStat(0);
    h_energy_vs_dt->SetStats(0);
    h_energy_vs_dt->GetXaxis()->SetTitle("dt (#mus)");
    h_energy_vs_dt->Draw("COLZ");
    c_evsdt->SaveAs((OUTPUT_DIR + "/Michel_Energy_vs_dt.png").c_str());
    canvases.push_back(c_evsdt);

    // Top Veto Muon
    TCanvas *c_top = new TCanvas("c_top", "Top Veto Muon", 800, 600);
    gStyle->SetOptStat(1111);
    h_top_vp_muon->SetLineColor(kMagenta);
    h_top_vp_muon->SetLineWidth(2);
    h_top_vp_muon->Draw();
    c_top->SaveAs((OUTPUT_DIR + "/Top_Veto_Muon.png").c_str());
    canvases.push_back(c_top);

    // Wide Side Veto Muon
    TCanvas *c_wide = new TCanvas("c_wide", "Wide Side Veto Muon", 800, 600);
    gStyle->SetOptStat(1111);
    h_wide_side_vp_muon->SetLineColor(kCyan);
    h_wide_side_vp_muon->SetLineWidth(2);
    h_wide_side_vp_muon->Draw();
    c_wide->SaveAs((OUTPUT_DIR + "/Wide_Side_Veto_Muon.png").c_str());
    canvases.push_back(c_wide);

    // Thin Side Veto Muon
    TCanvas *c_thin = new TCanvas("c_thin", "Thin Side Veto Muon", 800, 600);
    gStyle->SetOptStat(1111);
    h_thin_side_vp_muon->SetLineColor(kGreen);
    h_thin_side_vp_muon->SetLineWidth(2);
    h_thin_side_vp_muon->Draw();
    c_thin->SaveAs((OUTPUT_DIR + "/Thin_Side_Veto_Muon.png").c_str());
    canvases.push_back(c_thin);

    // Trigger Bits Distribution
    TCanvas *c_trigger = new TCanvas("c_trigger", "Trigger Bits", 800, 600);
    gStyle->SetOptStat(1111);
    h_trigger_bits->SetLineColor(kOrange);
    h_trigger_bits->SetLineWidth(2);
    h_trigger_bits->Draw();
    c_trigger->SaveAs((OUTPUT_DIR + "/TriggerBits_Distribution.png").c_str());
    canvases.push_back(c_trigger);

    // Good vs Bad Events
    TCanvas *c_goodbad = new TCanvas("c_goodbad", "Good vs Bad Events", 800, 600);
    gStyle->SetOptStat(1111);
    h_good_vs_bad->SetFillColor(kBlue);
    h_good_vs_bad->GetXaxis()->SetBinLabel(1, "Good Events");
    h_good_vs_bad->GetXaxis()->SetBinLabel(2, "Bad Events");
    h_good_vs_bad->Draw("BAR");
    c_goodbad->SaveAs((OUTPUT_DIR + "/Good_vs_Bad_Events.png").c_str());
    canvases.push_back(c_goodbad);

    // Peak Position RMS
    TCanvas *c_rms = new TCanvas("c_rms", "Peak Position RMS", 800, 600);
    gStyle->SetOptStat(1111);
    h_peak_position_rms->SetLineColor(kGreen);
    h_peak_position_rms->SetLineWidth(2);
    h_peak_position_rms->Draw();
    c_rms->SaveAs((OUTPUT_DIR + "/Peak_Position_RMS.png").c_str());
    canvases.push_back(c_rms);

    // Create veto panel plots and save to ROOT
    createVetoPanelPlots(h_veto_panel, OUTPUT_DIR, outFile);

    // Now write all histograms and canvases to ROOT file
    outFile->cd();
    
    // Write histograms
    h_muon_energy->Write();
    h_michel_energy->Write();
    h_dt_michel->Write();
    h_energy_vs_dt->Write();
    h_top_vp_muon->Write();
    h_wide_side_vp_muon->Write();
    h_thin_side_vp_muon->Write();
    h_trigger_bits->Write();
    h_good_vs_bad->Write();
    h_peak_position_rms->Write();
    for (int i = 0; i < 10; i++) {
        h_veto_panel[i]->Write();
    }
    
    // Write canvases to ROOT file (these contain the visual plots with fits, legends, etc.)
    const char* canvas_names[] = {
        "Muon_Energy", "Michel_Energy", "Michel_dt", "Michel_Energy_vs_dt",
        "Top_Veto_Muon", "Wide_Side_Veto_Muon", "Thin_Side_Veto_Muon",
        "TriggerBits_Distribution", "Good_vs_Bad_Events", "Peak_Position_RMS"
    };
    
    for (size_t i = 0; i < canvases.size() && i < 10; i++) {
        if (canvases[i]) {
            canvases[i]->Write(canvas_names[i]);
        }
    }
    
    outFile->Close();
    delete outFile;
    
    cout << "\n✓ All histograms and canvases saved to: " << OUTPUT_DIR << "/all_histograms.root" << endl;
    cout << "  The ROOT file contains:" << endl;
    cout << "  - All histograms (data)" << endl;
    cout << "  - All canvases (visual plots with fits, legends, etc.)" << endl;

    // Clean up
    delete h_muon_energy;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_top_vp_muon;
    delete h_wide_side_vp_muon;
    delete h_thin_side_vp_muon;
    delete h_trigger_bits;
    delete h_good_vs_bad;
    delete h_peak_position_rms;
    for (int i = 0; i < 10; i++) {
        delete h_veto_panel[i];
    }
    if (expFit) delete expFit;
    for (size_t i = 0; i < canvases.size(); i++) {
        delete canvases[i];
    }

    cout << "\n========================================" << endl;
    cout << "Analysis Complete!" << endl;
    cout << "========================================" << endl;
    cout << "Results saved in: " << OUTPUT_DIR << "/" << endl;
    cout << "\nSummary Statistics:" << endl;
    cout << "  Total Events Processed: " << num_events << endl;
    cout << "  Good Events: " << num_good_events << endl;
    cout << "  Muons Detected: " << num_muons << endl;
    cout << "  Michel Electrons Detected: " << num_michels << endl;
    cout << "\n✓ ROOT file with all histograms AND canvases: all_histograms.root" << endl;
    cout << "  - Histograms: 'muon_energy', 'michel_energy', 'DeltaT', etc." << endl;
    cout << "  - Canvases (with fits, legends): 'Muon_Energy', 'Michel_dt', etc." << endl;
    cout << "\nTo open the ROOT file and view canvases:" << endl;
    cout << "  root all_histograms.root" << endl;
    cout << "  TBrowser b" << endl;
    cout << "  Or: c1->Draw() to view individual canvases" << endl;
    cout << "========================================" << endl;

    return 0;
}
