#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TLatex.h>
#include <TF1.h>
#include <TCanvas.h>
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
#include <cstdlib>
#include <random>
#include <memory>

using std::cout;
using std::cerr;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0,10,7,2,6,3,8,9,11,4,5,1};
const int BS_UNCERTAINTY = 5;
const int EV61_THRESHOLD = 1200;
const double MUON_ENERGY_THRESHOLD = 50;
const double MICHEL_ENERGY_MIN = 40;
const double MICHEL_ENERGY_MAX = 1000;
const double MICHEL_ENERGY_MAX_DT = 500;
const double MICHEL_DT_MIN = 0.76;
const double MICHEL_DT_MAX = 16.0;
const int ADCSIZE = 45;
const double PEAK_POSITION_RMS_CUT = 2.5;
const double AREA_HEIGHT_RATIO_CUT = 1.2;

// Channel-specific trigger thresholds
const std::vector<double> TRIGGER_THRESHOLDS = {
   100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30,
    100
};

// Veto panel thresholds
const std::vector<double> TOP_VP_THRESHOLDS = {1000,1000}; // Channels 12-13 (ADC)
const std::vector<double> WIDE_SIDE_VP_THRESHOLDS = {1100, 1500, 1200, 1375}; // Channels 14-17 (ADC)
const std::vector<double> THIN_SIDE_VP_THRESHOLDS = {525, 700, 700, 500}; // Channels 18-21 (ADC)

// Fit ranges
const double FIT_MIN = 1.0;
const double FIT_MAX = 10.0;

// Generate unique output directory
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm tstruct;
    char buffer[20];
    tstruct = *localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tstruct);
    return string(buffer);
}
string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

// Pulse structures
struct pulse_temp {
    double start = 0;
    double end = 0;
    double peak = 0;
    double energy = 0;
    int peak_position = -1;
};

struct pulse {
    double start = 0;
    double end = 0;
    double peak = 0;
    double energy = 0;
    double number = 0;
    bool single = false;
    bool beam = false;
    double trigger = 0;
    double side_vp_energy = 0;
    double top_vp_energy = 0;
    double all_vp_energy = 0;
    double last_muon_time = 0;
    bool is_muon = false;
    bool is_michel = false;
    double peak_position_rms = 0;
    bool is_good_event = false;
};

// Michel candidate structure
struct MichelCandidate {
    double dt;
    double energy;
    int eventID;
    string fileName;
};

// Fitting functions
Double_t fitGauss(Double_t *x, Double_t *par) {
    return par[0] * TMath::Gaus(x[0], par[1], par[2]);
}

Double_t six_fit_func(Double_t *x, Double_t *par) {
    return par[0] * TMath::Gaus(x[0], par[1], par[2]) + 
           par[3] * TMath::Gaus(x[0], par[4], par[5]);
}

Double_t eight_fit_func(Double_t *x, Double_t *par) {
    return par[0] * TMath::Gaus(x[0], par[1], par[2]) + 
           par[3] * TMath::Gaus(x[0], par[4], par[5]) + 
           par[6] * TMath::Gaus(x[0], 2.0 * par[4], TMath::Sqrt(2.0 * par[5]*par[5] - par[2]*par[2])) + 
           par[7] * TMath::Gaus(x[0], 3.0 * par[4], TMath::Sqrt(3.0 * par[5]*par[5] - 2.0 * par[2]*par[2]));
}

Double_t ExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0]/par[1]) + par[2];
}

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
    return static_cast<double>(most_common);
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

// Create output directory
bool createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st)) {
        if (mkdir(dirName.c_str(), 0755)) {
            cerr << "Error: Could not create directory " << dirName << endl;
            return false;
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
    return true;
}

bool isGoodEvent(const std::vector<pulse_temp>& pmt_pulses, const Double_t* mu1, const Double_t* baselineRMS) {
    int countAbove2PE = 0;
    for (int pmt = 0; pmt < N_PMTS; pmt++) {
        if (pmt_pulses[pmt].peak > 0 && pmt_pulses[pmt].peak > 2 * mu1[pmt]) {
            countAbove2PE++;
        }
    }

    if (countAbove2PE >= 3) {
        vector<Double_t> peakPositions;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            if (pmt_pulses[pmt].peak > 0) {
                peakPositions.push_back(pmt_pulses[pmt].peak_position);
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
        int countConditionB = 0;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            if (pmt_pulses[pmt].peak > 0) {
                if (pmt_pulses[pmt].peak > 3 * baselineRMS[PMT_CHANNEL_MAP[pmt]] && 
                    (pmt_pulses[pmt].energy / pmt_pulses[pmt].peak) > AREA_HEIGHT_RATIO_CUT) {
                    countConditionB++;
                }
            }
        }

        if (countConditionB >= 3) {
            vector<Double_t> peakPositions;
            for (int pmt = 0; pmt < N_PMTS; pmt++) {
                if (pmt_pulses[pmt].peak > 0) {
                    peakPositions.push_back(pmt_pulses[pmt].peak_position);
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
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                               Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
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
            nLEDFlashes[pmt]++;
        }
    }

    Int_t defaultErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kError;

    TCanvas *c = new TCanvas("c", "SPE Fits", 1200, 800);
    c->Divide(4, 3);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);
    
    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        c->cd(i+1);
        
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
        histArea[i]->Fit(f8, "Q", "", -50, 400);

        mu1[i] = f8->GetParameter(4);
        mu1_err[i] = f8->GetParError(4);

        histArea[i]->Draw();
        f8->Draw("same");
        
        delete f1;
        delete f6;
        delete f8;
    }

    string plotName = OUTPUT_DIR + "/SPE_Fits.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved SPE plot: " << plotName << endl;

    gErrorIgnoreLevel = defaultErrorLevel;
    
    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]) delete histArea[i];
    }
    delete c;
    calibFile->Close();
    delete calibFile;
    return true;
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

    OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();
    if (!createOutputDirectory(OUTPUT_DIR)) {
        return 1;
    }

    // Validate veto threshold sizes
    if (TOP_VP_THRESHOLDS.size() != 2 || WIDE_SIDE_VP_THRESHOLDS.size() != 4 || THIN_SIDE_VP_THRESHOLDS.size() != 4) {
        cerr << "Error: Incorrect number of veto panel thresholds" << endl;
        return -1;
    }

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }

    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cerr << "Error: No input files found" << endl;
        return -1;
    }

    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    if (!performCalibration(calibFileName, mu1, mu1_err)) {
        cerr << "Calibration failed. Exiting..." << endl;
        return 1;
    }

    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    int total_events = 0;
    int total_good_events = 0;
    int total_muons = 0;
    int total_michels = 0;
    std::map<int, int> trigger_counts;

    // Define histograms
    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts/100 p.e.", 550, -500, 5000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/4 p.e.", 200, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaTInital", "Muon-Michel Time Difference;Time to Previous event(Muon)(#mus);Counts/0.25 #mus", 64, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 1000, 200, 0, 2000);
    TH1D* h_side_vp_muon = new TH1D("side_vp_muon", "Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 8000);
    TH1D* h_top_vp_muon = new TH1D("top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 2000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);
    TH1D* h_peak_position_rms = new TH1D("peak_position_rms", "Peak Position RMS Distribution;RMS (samples);Counts", 100, 0, 10);
    TH1D* h_good_vs_bad = new TH1D("good_vs_bad", "Event Quality;Quality;Counts", 2, 0, 2);

    // Veto panel histograms
    TH1D* h_veto_panel_energy[10];
    const string veto_panel_names[10] = {
        "Top_Veto_Panel_12", "Top_Veto_Panel_13",
        "Wide_Side_Veto_Panel_14", "Wide_Side_Veto_Panel_15",
        "Wide_Side_Veto_Panel_16", "Wide_Side_Veto_Panel_17",
        "Thin_Side_Veto_Panel_18", "Thin_Side_Veto_Panel_19",
        "Thin_Side_Veto_Panel_20", "Thin_Side_Veto_Panel_21"
    };
    for (int i = 0; i < 10; i++) {
        if (i < 2) { // Top veto panels (12-13)
            h_veto_panel_energy[i] = new TH1D(veto_panel_names[i].c_str(), 
                                             Form("%s;Energy (ADC);Counts", veto_panel_names[i].c_str()), 
                                             200, 0, 6000);
        } else if (i < 6) { // Wide side veto panels (14-17)
            h_veto_panel_energy[i] = new TH1D(veto_panel_names[i].c_str(), 
                                             Form("%s;Energy (ADC);Counts", veto_panel_names[i].c_str()), 
                                             200, 0, 6000);
        } else { // Thin side veto panels (18-21)
            h_veto_panel_energy[i] = new TH1D(veto_panel_names[i].c_str(), 
                                             Form("%s;Energy (ADC);Counts", veto_panel_names[i].c_str()), 
                                             200, 0, 4000);
        }
    }

    // Global tracking
    double last_muon_time = -1000.0;
    std::map<double, double> global_muons;
    std::set<double> global_muon_times_with_michel;
    std::vector<MichelCandidate> michel_candidates;

    for (const auto& inputFileName : inputFiles) {
        int file_events = 0;
        int file_good_events = 0;
        int file_muons = 0;
        int file_michels = 0;

        std::unique_ptr<TFile> f(TFile::Open(inputFileName.c_str()));
        if (!f || f->IsZombie()) {
            cout << "Error: Cannot open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TTree* t = dynamic_cast<TTree*>(f->Get("tree"));
        if (!t) {
            cout << "Error: Cannot find tree in file: " << inputFileName << endl;
            continue;
        }

        // Verify branches
        if (!t->GetBranch("eventID") || !t->GetBranch("nSamples") || !t->GetBranch("adcVal") ||
            !t->GetBranch("baselineMean") || !t->GetBranch("baselineRMS") || !t->GetBranch("pulseH") ||
            !t->GetBranch("peakPosition") || !t->GetBranch("area") || !t->GetBranch("nsTime") ||
            !t->GetBranch("triggerBits")) {
            cerr << "Error: Missing required branches in tree" << endl;
            continue;
        }

        Int_t eventID;
        Int_t nSamples[23];
        Short_t adcVal[23][45];
        Double_t baselineMean[23];
        Double_t baselineRMS[23];
        Double_t pulseH[23];
        Int_t peakPosition[23];
        Double_t area[23];
        Long64_t nsTime;
        Int_t triggerBits;

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
        cout << "Processing " << numEntries << " entries in " << inputFileName << endl;

        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            file_events++;
            total_events++;

            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;

            struct pulse p;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_vp_energy = 0;
            p.top_vp_energy = 0;
            p.all_vp_energy = 0;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;
            p.peak_position_rms = 0;
            p.is_good_event = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_vp_energy, top_vp_energy;
            std::vector<double> chan_starts_no_outliers;
            std::vector<pulse_temp> pmt_pulses(N_PMTS);
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> veto_energies(10, 0);

            // Set event time using nsTime and verify triggerBits
            p.start = nsTime / 1000.0;
            p.end = p.start;
            bool trigger_found = false;
            for (int iChan = 0; iChan < 23; iChan++) {
                if (triggerBits & (1 << iChan)) {
                    for (int i = 0; i < ADCSIZE; i++) {
                        double iBinContent = adcVal[iChan][i] - baselineMean[iChan];
                        if (iBinContent >= TRIGGER_THRESHOLDS[iChan]) {
                            trigger_found = true;
                            break;
                        }
                    }
                }
            }
            if (!trigger_found) {
                cout << "Warning: No channel with set trigger bit exceeded threshold in event " << eventID << endl;
            }

            // Process pulses
            for (int iChan = 0; iChan < 23; iChan++) {
                if (pulseH[iChan] < TRIGGER_THRESHOLDS[iChan]) continue;

                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        double iBinContent = h_wf.GetBinContent(iBin);
                        if (iBinContent >= TRIGGER_THRESHOLDS[iChan]) {
                            ev61_energy += iBinContent;
                        }
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBinContent >= TRIGGER_THRESHOLDS[iChan] && !onPulse) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < BS_UNCERTAINTY || iBin == ADCSIZE) {
                            pulse_temp pt;
                            pt.start = thresholdBin * 16.0 / 1000.0;
                            pt.peak = peak;
                            pt.end = iBin * 16.0 / 1000.0;
                            pt.peak_position = peakBin;
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            pt.energy = pulseEnergy;
                            
                            if (iChan <= 11) {
                                int pmt_idx = -1;
                                for (int k = 0; k < N_PMTS; k++) {
                                    if (PMT_CHANNEL_MAP[k] == iChan) {
                                        pmt_idx = k;
                                        break;
                                    }
                                }
                                if (pmt_idx >= 0) {
                                    pmt_pulses[pmt_idx] = pt;
                                    if (mu1[pmt_idx] > 0) {
                                        pt.energy /= mu1[pmt_idx];
                                        pt.peak /= mu1[pmt_idx];
                                    }
                                    all_chan_start.push_back(pt.start);
                                    all_chan_end.push_back(pt.end);
                                    all_chan_peak.push_back(pt.peak);
                                    all_chan_energy.push_back(pt.energy);
                                    if (pt.energy > 1) p.number += 1;
                                }
                            }
                            if (iChan >= 12 && iChan <= 13) { // Top veto panels
                                for (int j = thresholdBin; j <= iBin; j++) {
                                    allPulseEnergy += h_wf.GetBinContent(j);
                                }
                                top_vp_energy.push_back(allPulseEnergy);
                                veto_energies[iChan - 12] = allPulseEnergy;
                            } else if (iChan >= 14 && iChan <= 17) { // Wide side veto panels
                                for (int j = thresholdBin; j <= iBin; j++) {
                                    allPulseEnergy += h_wf.GetBinContent(j);
                                }
                                side_vp_energy.push_back(allPulseEnergy);
                                veto_energies[iChan - 12] = allPulseEnergy;
                            } else if (iChan >= 18 && iChan <= 21) { // Thin side veto panels
                                for (int j = thresholdBin; j <= iBin; j++) {
                                    allPulseEnergy += h_wf.GetBinContent(j);
                                }
                                side_vp_energy.push_back(allPulseEnergy);
                                veto_energies[iChan - 12] = allPulseEnergy;
                            }
                            peak = 0;
                            peakBin = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            if (!all_chan_end.empty()) {
                p.end = p.start + mostFrequent(all_chan_end);
            }
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0);
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0);
            p.side_vp_energy = std::accumulate(side_vp_energy.begin(), side_vp_energy.end(), 0.0);
            p.top_vp_energy = std::accumulate(top_vp_energy.begin(), top_vp_energy.end(), 0.0);
            p.all_vp_energy = p.side_vp_energy + p.top_vp_energy;

            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) {
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0);

            vector<Double_t> peakPositions;
            for (const auto& pt : pmt_pulses) {
                if (pt.peak > 0) {
                    peakPositions.push_back(pt.peak_position);
                }
            }
            if (!peakPositions.empty()) {
                Double_t dummyMean;
                CalculateMeanAndRMS(peakPositions, dummyMean, p.peak_position_rms);
                h_peak_position_rms->Fill(p.peak_position_rms);
            }

            p.is_good_event = isGoodEvent(pmt_pulses, mu1, baselineRMS);
            h_good_vs_bad->Fill(p.is_good_event ? 0 : 1);

            if (p.is_good_event) {
                file_good_events++;
                total_good_events++;

                bool veto_hit = false;
                for (size_t i = 0; i < 2; i++) { // Top veto panels (12-13)
                    if (veto_energies[i] > TOP_VP_THRESHOLDS[i]) {
                        veto_hit = true;
                        break;
                    }
                }
                for (size_t i = 2; i < 6; i++) { // Wide side veto panels (14-17)
                    if (veto_energies[i] > WIDE_SIDE_VP_THRESHOLDS[i - 2]) {
                        veto_hit = true;
                        break;
                    }
                }
                for (size_t i = 6; i < 10; i++) { // Thin side veto panels (18-21)
                    if (veto_energies[i] > THIN_SIDE_VP_THRESHOLDS[i - 6]) {
                        veto_hit = true;
                        break;
                    }
                }

                if ((p.energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                    (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                    p.is_muon = true;
                    last_muon_time = p.start;
                    global_muons[p.start] = p.energy;
                    file_muons++;
                    total_muons++;
                    h_side_vp_muon->Fill(p.side_vp_energy);
                    h_top_vp_muon->Fill(p.top_vp_energy);

                    // Fill veto panel histograms for this muon event
                    for (int i = 0; i < 10; i++) {
                        if ((i < 2 && veto_energies[i] > TOP_VP_THRESHOLDS[i]) ||
                            (i >= 2 && i < 6 && veto_energies[i] > WIDE_SIDE_VP_THRESHOLDS[i - 2]) ||
                            (i >= 6 && veto_energies[i] > THIN_SIDE_VP_THRESHOLDS[i - 6])) {
                            h_veto_panel_energy[i]->Fill(veto_energies[i]);
                        }
                    }
                }

                double dt = p.start - last_muon_time;
                bool veto_low = true;
                for (size_t i = 0; i < 2; i++) {
                    if (veto_energies[i] > TOP_VP_THRESHOLDS[i]) {
                        veto_low = false;
                        break;
                    }
                }
                for (size_t i = 2; i < 6; i++) {
                    if (veto_energies[i] > WIDE_SIDE_VP_THRESHOLDS[i - 2]) {
                        veto_low = false;
                        break;
                    }
                }
                for (size_t i = 6; i < 10; i++) {
                    if (veto_energies[i] > THIN_SIDE_VP_THRESHOLDS[i - 6]) {
                        veto_low = false;
                        break;
                    }
                }

                bool is_michel_candidate = p.energy >= MICHEL_ENERGY_MIN &&
                                          p.energy <= MICHEL_ENERGY_MAX &&
                                          dt >= MICHEL_DT_MIN &&
                                          dt <= MICHEL_DT_MAX &&
                                          p.number >= 10 &&
                                          veto_low &&
                                          p.trigger != 1 &&
                                          p.trigger != 4 &&
                                          p.trigger != 8 &&
                                          p.trigger != 16;

                bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

                if (is_michel_candidate) {
                    p.is_michel = true;
                    file_michels++;
                    total_michels++;
                    global_muon_times_with_michel.insert(last_muon_time);
                    h_michel_energy->Fill(p.energy);
                }

                if (is_michel_for_dt) {
                    MichelCandidate candidate;
                    candidate.dt = dt;
                    candidate.energy = p.energy;
                    candidate.eventID = eventID;
                    candidate.fileName = inputFileName;
                    michel_candidates.push_back(candidate);
                    h_dt_michel->Fill(dt);
                    h_energy_vs_dt->Fill(dt, p.energy);
                }
            }

            p.last_muon_time = last_muon_time;
        }

        cout << "File " << inputFileName << " Statistics:\n";
        cout << "Total Events: " << file_events << "\n";
        cout << "Good Events (passed cuts): " << file_good_events << "\n";
        cout << "Muons Detected: " << file_muons << "\n";
        cout << "Michel Electrons Detected: " << file_michels << "\n";
        cout << "------------------------\n";
    }

    // Fill muon energy histogram
    for (const auto& muon : global_muons) {
        if (global_muon_times_with_michel.find(muon.first) != global_muon_times_with_michel.end()) {
            h_muon_energy->Fill(muon.second);
        }
    }

    // Print triggerBits distribution
    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }

    // Save individual veto panel plots with default stats
    TCanvas *c_veto = new TCanvas("c_veto", "Veto Panel Energy Depositions", 800, 600);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);
    
    for (int i = 0; i < 10; i++) {
        c_veto->Clear();
        h_veto_panel_energy[i]->Draw();
        string vetoPlotName = OUTPUT_DIR + "/" + veto_panel_names[i] + ".png";
        c_veto->SaveAs(vetoPlotName.c_str());
        cout << "Saved veto panel plot: " << vetoPlotName << endl;
    }

    // Create combined veto panel canvas
    TCanvas *c_veto_combined = new TCanvas("c_veto_combined", "Combined Veto Panel Energy Depositions", 1200, 800);
    c_veto_combined->Divide(4, 3);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);
    
    for (int i = 0; i < 10; i++) {
        c_veto_combined->cd(i+1);
        h_veto_panel_energy[i]->Draw();
    }
    
    string vetoCombinedPlotName = OUTPUT_DIR + "/Combined_Veto_Panel_Energies.png";
    c_veto_combined->SaveAs(vetoCombinedPlotName.c_str());
    cout << "Saved combined veto panel plot: " << vetoCombinedPlotName << endl;
    
    delete c_veto;
    delete c_veto_combined;

    // Create and save other plots
    TCanvas *c = new TCanvas("c", "Analysis Plots", 800, 600);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    // Muon Energy
    c->Clear();
    h_muon_energy->Draw();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel Energy
    c->Clear();
    h_michel_energy->Draw();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel dt with exponential fit
    c->Clear();
    h_dt_michel->GetXaxis()->SetTitle("Time to previous event (Muon) (#mus)");
    h_dt_michel->Draw();

    if (h_dt_michel->GetEntries() > 5) {
        double integral = h_dt_michel->Integral(h_dt_michel->FindBin(FIT_MIN), h_dt_michel->FindBin(FIT_MAX));
        double bin_width = h_dt_michel->GetBinWidth(1);
        double N0_init = integral * bin_width / (FIT_MAX - FIT_MIN);
        double C_init = 0;
        int bin_14 = h_dt_michel->FindBin(14.0);
        int bin_16 = h_dt_michel->FindBin(16.0);
        double min_content = 1e9;
        for (int i = bin_14; i <= bin_16; i++) {
            double content = h_dt_michel->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        TH1D *h_log = (TH1D*)h_dt_michel->Clone("h_log");
        for (int i = 1; i <= h_log->GetNbinsX(); i++) {
            double content = h_log->GetBinContent(i);
            double error = h_log->GetBinError(i);
            if (content > C_init && error > 0) {
                h_log->SetBinContent(i, log(content - C_init));
                h_log->SetBinError(i, error / (content - C_init));
            } else {
                h_log->SetBinContent(i, 0);
                h_log->SetBinError(i, 0);
            }
        }
        TF1 *linearFit = new TF1("linearFit", "[0] - x/[1]", FIT_MIN, FIT_MAX);
        linearFit->SetParameters(log(N0_init), 2.2);
        h_log->Fit(linearFit, "Q", "", FIT_MIN, FIT_MAX);
        double tau_init = linearFit->GetParameter(1);
        delete linearFit;
        delete h_log;

        TF1 *expFit = new TF1("expFit", ExpFit, FIT_MIN, FIT_MAX, 3);
        expFit->SetParameters(N0_init, tau_init, C_init);
        expFit->SetParLimits(0, 0, N0_init * 100);
        expFit->SetParLimits(1, 0.1, 20.0);
        expFit->SetParLimits(2, -C_init * 10, C_init * 10);
        expFit->SetParNames("N_{0}", "#tau", "C");
        expFit->SetNpx(1000);

        h_dt_michel->Fit(expFit, "RE", "", FIT_MIN, FIT_MAX);
        expFit->Draw("same");

        cout << "Exponential Fit Results (Michel dt, " << FIT_MIN << "-" << FIT_MAX << " µs):\n";
        cout << Form("N₀ = %.1f ± %.1f", expFit->GetParameter(0), expFit->GetParError(0)) << endl;
        cout << Form("τ = %.4f ± %.4f µs", expFit->GetParameter(1), expFit->GetParError(1)) << endl;
        cout << Form("C = %.1f ± %.1f", expFit->GetParameter(2), expFit->GetParError(2)) << endl;
        cout << Form("χ²/NDF = %.4f", expFit->GetChisquare() / expFit->GetNDF()) << endl;
        cout << "----------------------------------------" << endl;

        delete expFit;
    } else {
        cout << "Warning: Insufficient entries (" << h_dt_michel->GetEntries() << ") for exponential fit." << endl;
    }

    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Energy vs dt
    c->Clear();
    h_energy_vs_dt->Draw("COLZ");
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Side Veto Muon
    c->Clear();
    h_side_vp_muon->Draw();
    plotName = OUTPUT_DIR + "/Side_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Top Veto Muon
    c->Clear();
    h_top_vp_muon->Draw();
    plotName = OUTPUT_DIR + "/Top_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Trigger Bits Distribution
    c->Clear();
    h_trigger_bits->Draw();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Peak Position RMS
    c->Clear();
    h_peak_position_rms->Draw();
    plotName = OUTPUT_DIR + "/Peak_Position_RMS.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Good vs Bad Events
    c->Clear();
    h_good_vs_bad->Draw("BAR");
    plotName = OUTPUT_DIR + "/Good_vs_Bad_Events.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Save histograms to ROOT file
    TFile *outFile = new TFile((OUTPUT_DIR + "/analysis_output.root").c_str(), "RECREATE");
    h_muon_energy->Write();
    h_michel_energy->Write();
    h_dt_michel->Write();
    h_energy_vs_dt->Write();
    h_side_vp_muon->Write();
    h_top_vp_muon->Write();
    h_trigger_bits->Write();
    h_peak_position_rms->Write();
    h_good_vs_bad->Write();
    for (int i = 0; i < 10; i++) {
        h_veto_panel_energy[i]->Write();
    }
    outFile->Close();
    delete outFile;

    // Clean up
    delete h_muon_energy;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_vp_muon;
    delete h_top_vp_muon;
    delete h_trigger_bits;
    delete h_peak_position_rms;
    delete h_good_vs_bad;
    for (int i = 0; i < 10; i++) {
        delete h_veto_panel_energy[i];
    }
    delete c;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (*.png, analysis_output.root)" << endl;
    cout << "Total Events Processed: " << total_events << endl;
    cout << "Total Good Events: " << total_good_events << endl;
    cout << "Total Muons Detected: " << total_muons << endl;
    cout << "Total Michel Electrons Detected: " << total_michels << endl;
    return 0;
}
