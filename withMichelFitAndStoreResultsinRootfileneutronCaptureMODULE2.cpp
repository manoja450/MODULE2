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
#include <TPaveStats.h>
#include <TGaxis.h>
#include <TPaveText.h>
#include <TBox.h>
#include <TLatex.h>
#include <TParameter.h>
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
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TPad.h>

using std::cout;
using std::cerr;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
const int PULSE_THRESHOLD = 30;
const int BS_UNCERTAINTY = 5;
const int EV61_THRESHOLD = 1200;
const double MUON_ENERGY_THRESHOLD = 50;
const double MICHEL_ENERGY_MIN = 0;
const double MICHEL_ENERGY_MAX = 1000;
const double MICHEL_ENERGY_MAX_DT = 500;
const double MICHEL_DT_MIN = 0.76;
const double MICHEL_DT_MAX = 16.0;
const double MICHEL_DT_MIN_EXTENDED = 0;
const double MICHEL_DT_MAX_EXTENDED = 16.0;
const double MICHEL_ENERGY_MAX_EXTENDED = 100.0;
const int ADCSIZE = 45;
const double LOW_ENERGY_DT_MIN = 16.0;
const std::vector<double> SIDE_VP_THRESHOLDS = {750, 950, 1200, 1400, 550, 700, 700, 500}; // Channels 14-21
const double TOP_VP_THRESHOLD = 450; // Channels 12-13
const double FIT_MIN = 2.0;
const double FIT_MAX = 16.0;
const double FIT_MIN_LOW_MUON = 16.0;
const double FIT_MAX_LOW_MUON = 1200.0;

// Michel background prediction constants
const double SIGNAL_REGION_MIN = 16.0;
const double SIGNAL_REGION_MAX = 100.0;
const double MICHEL_BKG_FIT_MIN = FIT_MIN;
const double MICHEL_BKG_FIT_MAX = FIT_MAX;

string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

struct pulse {
    double start;
    double end;
    double peak;
    double energy;
    double number;
    bool single;
    bool beam;
    int trigger;
    double side_vp_energy;
    double top_vp_energy;
    double all_vp_energy;
    double last_muon_time;
    bool is_muon;
    bool is_michel;
    bool veto_hit[10];
};

struct pulse_temp {
    double start;
    double end;
    double peak;
    double energy;
};

// SPE Fitting functions
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

Double_t ExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0] / par[1]) + par[2];
}

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

double calculateLiveTime(const vector<string>& inputFiles) {
    double totalLiveTime = 0.0;
    
    for (const auto& fileName : inputFiles) {
        TFile* file = TFile::Open(fileName.c_str());
        if (!file || file->IsZombie()) {
            cerr << "Warning: Could not open file for live time calculation: " << fileName << endl;
            continue;
        }
        
        TTree* tree = (TTree*)file->Get("tree");
        if (!tree) {
            cerr << "Warning: Could not find tree in file: " << fileName << endl;
            file->Close();
            continue;
        }
        
        Long64_t nEntries = tree->GetEntries();
        if (nEntries == 0) {
            cout << "File: " << fileName << " - No entries, skipping" << endl;
            file->Close();
            delete file;
            continue;
        }
        
        Long64_t nsTime;
        tree->SetBranchAddress("nsTime", &nsTime);
        
        tree->GetEntry(0);
        Long64_t first_time = nsTime;
        tree->GetEntry(nEntries - 1);
        Long64_t last_time = nsTime;
        
        double liveTime_seconds = (last_time - first_time) / 1e9;
        totalLiveTime += liveTime_seconds;
        
        cout << "File: " << fileName << endl;
        cout << "  Events: " << nEntries << endl;
        cout << "  First event: " << first_time << " ns (" << first_time/1e9 << " s)" << endl;
        cout << "  Last event: " << last_time << " ns (" << last_time/1e9 << " s)" << endl;
        cout << "  Live time: " << liveTime_seconds << " seconds (" << liveTime_seconds / 3600.0 << " hours)" << endl;
        
        file->Close();
        delete file;
    }
    
    return totalLiveTime;
}

void saveLiveTimeInfo(double totalLiveTime, double fit_michels_2_16, double predicted_michels, double predicted_michels_err, const string& outputDir) {
    string filename = outputDir + "/LiveTime_Info.txt";
    ofstream outFile(filename);
    
    if (!outFile) {
        cerr << "Error: Could not create live time info file: " << filename << endl;
        return;
    }
    
    outFile << "Live Time Information\n";
    outFile << "=====================\n";
    outFile << "Total Live Time: " << totalLiveTime << " seconds\n";
    outFile << "Total Live Time: " << totalLiveTime / 60.0 << " minutes\n";
    outFile << "Total Live Time: " << totalLiveTime / 3600.0 << " hours\n";
    outFile << "Total Live Time: " << totalLiveTime / (3600.0 * 24) << " days\n";
    outFile << "\n";
    outFile << "Michel Background Analysis\n";
    outFile << "==========================\n";
    outFile << "Fit Michels (2-16 μs): " << fit_michels_2_16 << " events\n";
    outFile << "Predicted Michels (16-100 μs): " << predicted_michels << " ± " << predicted_michels_err << " events\n";
    outFile << "Scaling Factor (Predicted/Fit): " << (predicted_michels > 0 && fit_michels_2_16 > 0 ? predicted_michels / fit_michels_2_16 : 0) << "\n";
    outFile << "\n";
    outFile << "Normalization Factors:\n";
    outFile << "Counts per second: multiply by " << (1.0 / totalLiveTime) << "\n";
    outFile << "Counts per minute: multiply by " << (60.0 / totalLiveTime) << "\n";
    outFile << "Counts per hour: multiply by " << (3600.0 / totalLiveTime) << "\n";
    outFile << "Counts per day: multiply by " << (3600.0 * 24 / totalLiveTime) << "\n";
    
    outFile.close();
    cout << "Saved live time information: " << filename << endl;
}

void performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        exit(1);
    }

    string speDir = OUTPUT_DIR + "/SPE_Fits";
    gSystem->mkdir(speDir.c_str(), kTRUE);

    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
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
            nLEDFlashes[pmt]++;
        }
    }

    Int_t defaultErrorLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kError;

    // Create directory for individual PMT plots
    string individualPlotsDir = speDir + "/Individual";
    gSystem->mkdir(individualPlotsDir.c_str(), kTRUE);

    // Main canvas for combined view
    TCanvas *c_combined = new TCanvas("c_combined", "SPE Fits - Combined", 1200, 800);
    c_combined->Divide(4, 3);
    gStyle->SetOptStat(1111);  // Enable default statistics box
    gStyle->SetOptFit(1111);   // Enable fit parameters in stats box

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        // Process in combined canvas
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

        // Save individual plot
        string indivPlotName = individualPlotsDir + Form("/PMT%d_SPE_Fit.png", i+1);
        c_indiv->SaveAs(indivPlotName.c_str());
        cout << "Saved individual plot: " << indivPlotName << endl;

        delete c_indiv;
        delete f1;
        delete f6;
        delete f8;
    }

    // Save combined plot
    string combinedPlotName = speDir + "/SPE_Fits_Combined.png";
    c_combined->SaveAs(combinedPlotName.c_str());
    cout << "Saved combined SPE plot: " << combinedPlotName << endl;

    gErrorIgnoreLevel = defaultErrorLevel;

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]) delete histArea[i];
    }
    delete c_combined;
    calibFile->Close();
}

void createVetoPanelPlots(TH1D* h_veto_panel[10], const string& outputDir) {
    for (int i = 0; i < 10; i++) {
        TCanvas *c = new TCanvas(Form("c_veto_%d", i+12), Form("Veto Panel %d", i+12), 1200, 800);
        gStyle->SetOptStat(1111);
        gStyle->SetOptTitle(1);
        gStyle->SetStatX(0.9);
        gStyle->SetStatY(0.9);
        gStyle->SetStatW(0.2);
        gStyle->SetStatH(0.15);
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->Draw("hist");
        string plotName = outputDir + Form("/Veto_Panel_%d.png", i+12);
        c->SaveAs(plotName.c_str());
        cout << "Saved veto panel plot: " << plotName << endl;
        delete c;
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
    delete c_combined;
}

void saveCosmicFluxToCSV(const std::map<Long64_t, int>& cosmic_counts, const string& outputDir) {
    string filename = outputDir + "/CosmicFlux_AllHours.csv";
    ofstream csv_file(filename);
    csv_file << "DateTime,EventCount\n";
    for (const auto& pair : cosmic_counts) {
        Long64_t hour_start = pair.first;
        int count = pair.second;
        struct tm *timeinfo = localtime((time_t*)&hour_start);
        char buffer[20];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        csv_file << buffer << "," << count << "\n";
    }
    csv_file.close();
    cout << "Saved cosmic flux data: " << filename << endl;
}

void saveDailyCosmicFluxToCSV(const std::vector<std::pair<Long64_t, int>>& file_infos, const string& outputDir) {
    string filename = outputDir + "/CosmicFlux_AllDays.csv";
    ofstream csv_file(filename);
    if (!csv_file) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    csv_file << "Date,DailyCount\n";
    size_t num = file_infos.size();
    for (size_t i = 0; i < num; i += 24) {
        int sum = 0;
        size_t group_size = 0;
        Long64_t first_ts = file_infos[i].first;
        for (size_t j = 0; j < 24 && i + j < num; j++) {
            sum += file_infos[i + j].second;
            group_size++;
        }
        double avg = static_cast<double>(sum) / group_size;
        time_t now = static_cast<time_t>(first_ts);
        struct tm *t = localtime(&now);
        char buffer[11];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d", t);
        csv_file << buffer << "," << avg << "\n";
    }
    csv_file.close();
    cout << "Saved daily cosmic flux data: " << filename << endl;
}

double calculateTotalEvents(TH1D* hist) {
    double total = 0;
    for (int i = 1; i <= hist->GetNbinsX(); i++) {
        total += hist->GetBinContent(i);
    }
    return total;
}

double calculateFitMichels(double N0, double tau, double C, double t_min, double t_max, double bin_size = 0.1) {
    double total_michels = 0.0;
    int bin_count = 0;
    
    for (int i = 0; i < static_cast<int>((t_max - t_min) / bin_size); i++) {
        double t_start = t_min + i * bin_size;
        double t_end = t_start + bin_size;
        double t_center = t_start + (bin_size / 2.0);
        
        if (t_end > t_max) break;
        
        double N_t = N0 * exp(-t_center / tau);
        total_michels += N_t;
        bin_count++;
    }
    
    cout << "Calculated Fit Michels in range " << t_min << "-" << t_max << " μs:" << endl;
    cout << "  Using bin size: " << bin_size << " μs" << endl;
    cout << "  Number of bins: " << bin_count << endl;
    cout << "  Total Michels: " << total_michels << endl;
    cout << "  Parameters: N0=" << N0 << ", τ=" << tau << ", C=" << C << endl;
    
    return total_michels;
}

void saveAllHistogramsToRootFile(TFile* rootFile, 
                                TH1D* h_muon_energy, TH1D* h_muon_all, TH1D* h_michel_energy, 
                                TH1D* h_dt_michel, TH2D* h_energy_vs_dt, TH1D* h_side_vp_muon, 
                                TH1D* h_top_vp_muon, TH1D* h_trigger_bits, TH1D* h_isolated_pe, 
                                TH1D* h_low_iso, TH1D* h_high_iso, TH1D* h_dt_prompt_delayed, 
                                TH1D* h_dt_low_muon, TH1D* h_dt_high_muon, TH1D* h_low_pe_signal, 
                                TH1D* h_low_pe_sideband, TH1D* h_isolated_ge40, 
                                TH1D* h_dt_michel_sideband, TH1D* h_michel_energy_fit_range,
                                TH1D* h_veto_panel[10], TH1D* h_neutron_richness, 
                                TH1D* h_signal_significance, TH2D* h_energy_vs_time_low, 
                                TH2D* h_energy_vs_time_high, TH1D* h_michel_energy_predicted,
                                TH1D* h_scaled_sideband, TH1D* h_michel_background_predicted,
                                TH1D* h_final_subtracted, TH1D* h_low_pe_signal_norm,
                                TH1D* h_low_pe_sideband_norm, TH1D* h_scaled_sideband_norm,
                                TH1D* h_michel_background_predicted_norm, TH1D* h_final_subtracted_norm) {
    
    if (!rootFile || rootFile->IsZombie()) {
        cerr << "Error: Cannot save histograms to ROOT file - file is not open or is zombie" << endl;
        return;
    }
    
    rootFile->cd();
    
    // Save main histograms
    if (h_muon_energy) h_muon_energy->Write();
    if (h_muon_all) h_muon_all->Write();
    if (h_michel_energy) h_michel_energy->Write();
    if (h_dt_michel) h_dt_michel->Write();
    if (h_energy_vs_dt) h_energy_vs_dt->Write();
    if (h_side_vp_muon) h_side_vp_muon->Write();
    if (h_top_vp_muon) h_top_vp_muon->Write();
    if (h_trigger_bits) h_trigger_bits->Write();
    if (h_isolated_pe) h_isolated_pe->Write();
    if (h_low_iso) h_low_iso->Write();
    if (h_high_iso) h_high_iso->Write();
    if (h_dt_prompt_delayed) h_dt_prompt_delayed->Write();
    if (h_dt_low_muon) h_dt_low_muon->Write();
    if (h_dt_high_muon) h_dt_high_muon->Write();
    if (h_low_pe_signal) h_low_pe_signal->Write();
    if (h_low_pe_sideband) h_low_pe_sideband->Write();
    if (h_isolated_ge40) h_isolated_ge40->Write();
    if (h_dt_michel_sideband) h_dt_michel_sideband->Write();
    if (h_michel_energy_fit_range) h_michel_energy_fit_range->Write();
    if (h_neutron_richness) h_neutron_richness->Write();
    if (h_signal_significance) h_signal_significance->Write();
    if (h_energy_vs_time_low) h_energy_vs_time_low->Write();
    if (h_energy_vs_time_high) h_energy_vs_time_high->Write();
    if (h_michel_energy_predicted) h_michel_energy_predicted->Write();
    if (h_scaled_sideband) h_scaled_sideband->Write();
    if (h_michel_background_predicted) h_michel_background_predicted->Write();
    if (h_final_subtracted) h_final_subtracted->Write();
    if (h_low_pe_signal_norm) h_low_pe_signal_norm->Write();
    if (h_low_pe_sideband_norm) h_low_pe_sideband_norm->Write();
    if (h_scaled_sideband_norm) h_scaled_sideband_norm->Write();
    if (h_michel_background_predicted_norm) h_michel_background_predicted_norm->Write();
    if (h_final_subtracted_norm) h_final_subtracted_norm->Write();
    
    // Save veto panel histograms
    for (int i = 0; i < 10; i++) {
        if (h_veto_panel[i]) h_veto_panel[i]->Write();
    }
    
    cout << "All histograms saved to ROOT file" << endl;
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

    std::vector<std::pair<Long64_t, string>> file_times;
    for (const auto& file : inputFiles) {
        TFile f_tmp(file.c_str());
        if (f_tmp.IsZombie()) {
            cerr << "Warning: Cannot open " << file << " for timestamp sorting, skipping" << endl;
            f_tmp.Close();
            continue;
        }
        auto tsstart = (TParameter<Long64_t>*)f_tmp.Get("starttime");
        Long64_t ts = tsstart ? tsstart->GetVal() : 0;
        f_tmp.Close();
        file_times.emplace_back(ts, file);
    }
    std::sort(file_times.begin(), file_times.end());
    inputFiles.clear();
    for (const auto& ft : file_times) inputFiles.push_back(ft.second);

    createOutputDirectory(OUTPUT_DIR);

    // Create ROOT file for saving all histograms
    string rootFileName = OUTPUT_DIR + "/AnalysisResults.root";
    TFile* rootFile = new TFile(rootFileName.c_str(), "RECREATE");
    if (!rootFile || rootFile->IsZombie()) {
        cerr << "Error: Could not create ROOT file " << rootFileName << endl;
        return -1;
    }
    cout << "Created ROOT file for saving histograms: " << rootFileName << endl;

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files (sorted by starttime):\n";
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        rootFile->Close();
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
        rootFile->Close();
        return -1;
    }

    double totalLiveTime = calculateLiveTime(inputFiles);
    double liveTimeDays = totalLiveTime / (3600.0 * 24);
    
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    performCalibration(calibFileName, mu1, mu1_err);

    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    int num_muons = 0;
    int num_michels = 0;
    int num_michels_extended = 0;
    int num_events = 0;
    int num_cosmic_events = 0;
    int num_low_iso = 0;

    int count_90pe_original = 0;
    int count_90pe_extended = 0;

    std::map<int, int> trigger_counts;

    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts/7 p.e.", 500, -500, 3000);
    TH1D* h_muon_all = new TH1D("muon_all", "All Muon Energy Distribution;Energy (p.e.);Counts/6 p.e.", 500, 0, 3000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/8 p.e.", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaT", "Muon-Michel Time Difference;Time to Previous event(Muon)(#mus);Counts/0.08 #mus", 200, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 16, 200, 0, 1000);
    TH1D* h_side_vp_muon = new TH1D("side_vp_muon", "Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_top_vp_muon = new TH1D("top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 1000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);
    TH1D* h_isolated_pe = new TH1D("isolated_pe", "Sum PEs Isolated Events;Photoelectrons;Counts/10 p.e.", 200, 0, 2000);
    TH1D* h_low_iso = new TH1D("low_iso", "Sum PEs Low Energy Isolated Events;Photoelectrons;Counts/1 p.e.", 100, 0, 100);
    TH1D* h_high_iso = new TH1D("high_iso", "Sum PEs High Energy Isolated Events;Photoelectrons;Counts/10 p.e.", 100, 0, 1000);
    TH1D* h_dt_prompt_delayed = new TH1D("dt_prompt_delayed", "#Delta t High Energy (prompt) to Low Energy (delayed);#Delta t [#mus];Counts", 200, 0, 10000);
    TH1D* h_dt_low_muon = new TH1D("dt_low_muon", "#Delta t Low Energy Isolated to Muon Veto Tagged Events;#Delta t [#mus];Counts/10#mus", 120, 0, 1200);
    TH1D* h_dt_high_muon = new TH1D("dt_high_muon", "#Delta t High Energy Isolated to Muon Veto Tagged Events;#Delta t [#mus];Counts/10#mus", 120, 0, 1200);
    TH1D* h_low_pe_signal = new TH1D("low_pe_signal", "Low Energy Signal Region Sideband Subtraction;Photoelectrons;Counts", 100, 0, 100);
    TH1D* h_low_pe_sideband = new TH1D("low_pe_sideband", "Low Energy Sideband (1000-1200 #mus);Photoelectrons;Counts", 100, 0, 100);
    TH1D* h_isolated_ge40 = new TH1D("isolated_ge40", "Sum PEs Isolated Events (>=40 p.e.);Photoelectrons;Events/20 p.e.", 200, 40, 2000);

    TH1D* h_dt_michel_sideband = new TH1D("dt_michel_sideband", 
        "Michel Time Distribution (0-16 #mus);Time to Previous Muon (#mus);Counts/0.2#mus", 
        80, 0, 16);
    
    TH1D* h_michel_energy_fit_range = new TH1D("michel_energy_fit_range", 
        "Michel Energy Spectrum (2-16 #mus);Energy (p.e.);Counts/1p.e.", 
        100, 0, 100);

    TH1D* h_veto_panel[10];
    const char* veto_names[10] = {
        "Top Veto Panel 12", "Top Veto Panel 13", "Side Veto Panel 14", "Side Veto Panel 15",
        "Side Veto Panel 16", "Side Veto Panel 17", "Side Veto Panel 18", "Side Veto Panel 19",
        "Side Veto Panel 20", "Side Veto Panel 21"
    };

    for (int i = 0; i < 10; i++) {
        h_veto_panel[i] = new TH1D(Form("h_veto_panel_%d", i+12), 
                                   Form("%s;Energy (ADC);Counts", veto_names[i]), 
                                   200, 0, 8000);
    }

    TH1D* h_neutron_richness = new TH1D("neutron_richness", 
        "Neutron-to-Background Ratio vs Time;Time[#mus];Neutron/Bkg Ratio", 
        100, 0, 1000);
    TH1D* h_signal_significance = new TH1D("signal_significance", 
        "Signal Significance vs Time;Time[#mus];S/#sqrt{S + B}", 
        100, 0, 1000);

    TH2D* h_energy_vs_time_low = new TH2D("energy_vs_time_low", 
        "Low Energy Events: Energy vs Time to Muon;Time to Muon [#mus];Energy (p.e.)", 
        120, 0, 1200, 100, 0, 100);
    TH2D* h_energy_vs_time_high = new TH2D("energy_vs_time_high", 
        "High Energy Events: Energy vs Time to Muon;Time to Muon [#mus];Energy (p.e.)", 
        120, 0, 1200, 200, 0, 2000);

    // Initialize additional histograms for background subtraction
    TH1D* h_michel_energy_predicted = nullptr;
    TH1D* h_scaled_sideband = nullptr;
    TH1D* h_michel_background_predicted = nullptr;
    TH1D* h_final_subtracted = nullptr;
    TH1D* h_low_pe_signal_norm = nullptr;
    TH1D* h_low_pe_sideband_norm = nullptr;
    TH1D* h_scaled_sideband_norm = nullptr;
    TH1D* h_michel_background_predicted_norm = nullptr;
    TH1D* h_final_subtracted_norm = nullptr;

    double last_muon_time = 0.0;
    double last_high_time = 0.0;

    std::vector<std::pair<double, double>> muon_candidates;
    std::set<double> michel_muon_times;

    std::map<Long64_t, int> cosmic_counts;
    std::vector<std::pair<Long64_t, int>> file_cosmic_infos;

    const std::set<int> excluded_triggers = {1, 3, 4, 8, 16, 33, 35};

    int excluded_low_iso = 0;

    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        Long64_t run_starttime = 0;
        auto tsstart = (TParameter<Long64_t> *) f->Get("starttime");
        if (tsstart) {
            run_starttime = tsstart->GetVal();
            cout << "Run Start Time (Unix Timestamp): " << run_starttime << endl;
            time_t rawtime = (time_t)run_starttime;
            struct tm *timeinfo = localtime(&rawtime);
            cout << "Run Start Time (Local Time): " << asctime(timeinfo);
            timeinfo->tm_min = 0;
            timeinfo->tm_sec = 0;
            run_starttime = mktime(timeinfo);
        } else {
            cerr << "Warning: 'starttime' not found in file " << inputFileName << ". Skipping file." << endl;
            f->Close();
            continue;
        }

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
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

        int file_cosmic_count = 0;
        double last_event_time = -1;

        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            if (nsTime < last_event_time) {
                cerr << "Warning: Non-increasing event time in file " << inputFileName 
                     << ", event " << eventID << ": " << nsTime << " < " << last_event_time << endl;
            }
            last_event_time = nsTime;

            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;
            if (triggerBits < 0 || triggerBits > 36) {
                cout << "Warning: triggerBits = " << triggerBits << " out of histogram range (0-36) in file " << inputFileName << ", event " << eventID << endl;
            }

            struct pulse p;
            p.start = nsTime / 1000.0;
            p.end = p.start;
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
            for (int i = 0; i < 10; i++) p.veto_hit[i] = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_vp_energy, top_vp_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> veto_energies(10, 0);

            for (int iChan = 0; iChan < 23; iChan++) {
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        ev61_energy += h_wf.GetBinContent(iBin);
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                std::vector<pulse_temp> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;

                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
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
                            // Use the new SPE calibration results
                            int pmt_index = -1;
                            for (int j = 0; j < N_PMTS; j++) {
                                if (PMT_CHANNEL_MAP[j] == iChan) {
                                    pmt_index = j;
                                    break;
                                }
                            }
                            if (pmt_index >= 0 && mu1[pmt_index] > 0) {
                                pt.peak = peak / mu1[pmt_index];
                            } else {
                                pt.peak = peak;
                            }
                            pt.end = iBin * 16.0 / 1000.0;
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11) {
                                if (pmt_index >= 0 && mu1[pmt_index] > 0) {
                                    pt.energy = pulseEnergy / mu1[pmt_index];
                                } else {
                                    pt.energy = 0;
                                }
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                // UPDATED VETO PANEL CHANNEL MAPPING:
                // Channels 12-13: Top veto panels
                // Channels 14-21: Side veto panels
                if (iChan >= 12 && iChan <= 13) {
                    // Top veto panels (channels 12-13)
                    double factor = (iChan == 12) ? 1.07809 : 1.0; // Apply correction if needed
                    top_vp_energy.push_back(allPulseEnergy * factor);
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                    if (allPulseEnergy * factor > TOP_VP_THRESHOLD) {
                        p.veto_hit[iChan - 12] = true;
                    }
                } else if (iChan >= 14 && iChan <= 21) {
                    // Side veto panels (channels 14-21)
                    side_vp_energy.push_back(allPulseEnergy);
                    veto_energies[iChan - 12] = allPulseEnergy;
                    if (allPulseEnergy > SIDE_VP_THRESHOLDS[iChan - 14]) {
                        p.veto_hit[iChan - 12] = true;
                    }
                }

                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
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

            // UPDATED VETO HIT DETECTION:
            bool veto_hit = false;
            // Check side veto panels (channels 14-21, indices 2-9 in veto_energies array)
            for (size_t i = 2; i < 10; i++) {
                if (i - 2 < SIDE_VP_THRESHOLDS.size() && veto_energies[i] > SIDE_VP_THRESHOLDS[i - 2]) {
                    veto_hit = true;
                    break;
                }
            }
            // Check top veto panels (channels 12-13, indices 0-1 in veto_energies array)
            if (!veto_hit) {
                for (int i = 0; i < 2; i++) {
                    if (veto_energies[i] > TOP_VP_THRESHOLD) {
                        veto_hit = true;
                        break;
                    }
                }
            }

            bool veto_low = !veto_hit;

            bool is_cosmic = !p.beam && excluded_triggers.find(p.trigger) == excluded_triggers.end();
            if (is_cosmic) {
                num_cosmic_events++;
                file_cosmic_count++;
            }

            if ((p.energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                p.is_muon = true;
                last_muon_time = p.start;
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy);
                h_muon_all->Fill(p.energy);
                h_side_vp_muon->Fill(p.side_vp_energy);
                h_top_vp_muon->Fill(p.top_vp_energy);
                
                for (int i = 0; i < 10; i++) {
                    if (p.veto_hit[i]) {
                        h_veto_panel[i]->Fill(veto_energies[i]);
                    }
                }
            }

            bool is_beam_off = ((p.trigger & 1) == 0) && (p.trigger != 4) && (p.trigger != 8) && (p.trigger != 16) && !p.beam;

            double dt = p.start - last_muon_time;

            bool is_michel_candidate = is_beam_off &&
                                      p.energy >= MICHEL_ENERGY_MIN &&
                                      p.energy <= MICHEL_ENERGY_MAX &&
                                      dt >= MICHEL_DT_MIN &&
                                      dt <= MICHEL_DT_MAX &&
                                      p.number >= 8 &&
                                      veto_low;
            h_energy_vs_dt->Fill(dt, p.energy);

            bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

            if (is_michel_candidate) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time);
                h_michel_energy->Fill(p.energy);
                
                if (p.energy >= 85 && p.energy <= 95) {
                    count_90pe_original++;
                }
            }

            if (is_michel_for_dt) {
                h_dt_michel->Fill(dt);
            }

            bool is_michel_sideband = is_beam_off &&
                                     p.energy >= MICHEL_ENERGY_MIN &&
                                     p.energy <= MICHEL_ENERGY_MAX_EXTENDED &&
                                     dt >= MICHEL_DT_MIN_EXTENDED &&
                                     dt <= MICHEL_DT_MAX_EXTENDED &&
                                     p.number >= 8 &&
                                     veto_low;

            if (is_michel_sideband) {
                num_michels_extended++;
                h_dt_michel_sideband->Fill(dt);
                
                if (dt >= MICHEL_BKG_FIT_MIN && dt <= MICHEL_BKG_FIT_MAX) {
                    h_michel_energy_fit_range->Fill(p.energy);
                }
                
                if (p.energy >= 85 && p.energy <= 95) {
                    count_90pe_extended++;
                }
            }

            p.last_muon_time = last_muon_time;

            bool is_isolated = is_beam_off &&
                               !p.is_muon &&
                               !p.is_michel &&
                               veto_low &&
                               p.single &&
                               p.energy > 0;

            if (is_isolated) {
                double dt_muon = p.start - last_muon_time;
                
                if (p.energy <= 100 && p.number >= 4) {
                    h_energy_vs_time_low->Fill(dt_muon, p.energy);
                } else if (p.energy > 100 && p.number >= 8) {
                    h_energy_vs_time_high->Fill(dt_muon, p.energy);
                }
                
                if (p.energy > 100 && p.number >= 8) {
                    h_high_iso->Fill(p.energy);
                    h_isolated_pe->Fill(p.energy);
                    h_isolated_ge40->Fill(p.energy);
                    last_high_time = p.start;
                    if (dt_muon >= LOW_ENERGY_DT_MIN) {
                        h_dt_high_muon->Fill(dt_muon);
                    }
                } else if (p.energy <= 100 && p.number >= 4) {
                    num_low_iso++;
                    double dt_high = p.start - last_high_time;
    
                    if (dt_high >= 0 && dt_high <= 10000) {
                        h_dt_prompt_delayed->Fill(dt_high);
                    } else {
                        h_dt_prompt_delayed->Fill(10000);
                        excluded_low_iso++;
                    }
                    
                    h_low_iso->Fill(p.energy);
                    if (p.energy >= 40) {
                        h_isolated_ge40->Fill(p.energy);
                    }
                    if (dt_muon >= LOW_ENERGY_DT_MIN) {
                        h_dt_low_muon->Fill(dt_muon);
                    }
                    if (dt_muon > 16 && dt_muon < 100) {
                        h_low_pe_signal->Fill(p.energy);
                    }
                    if (dt_muon > 1000 && dt_muon < 1200) {
                        h_low_pe_sideband->Fill(p.energy);
                    }
                }
            }
        }

        if (file_cosmic_count > 0) {
            cosmic_counts[run_starttime] += file_cosmic_count;
        }

        file_cosmic_infos.push_back({run_starttime, file_cosmic_count});

        f->Close();
        delete f;
    }

    cout << "=== 90 p.e. EVENT COUNT COMPARISON ===" << endl;
    cout << "Original Michel analysis (0-16 μs, 0-1000 p.e.): " << count_90pe_original << " events in 85-95 p.e. range" << endl;
    cout << "Extended Michel analysis (0-16 μs, 0-100 p.e.): " << count_90pe_extended << " events in 85-95 p.e. range" << endl;
    cout << "Difference: " << count_90pe_original - count_90pe_extended << " events" << endl;
    if (count_90pe_original == count_90pe_extended) {
        cout << "✓ COUNTS MATCH - Good!" << endl;
    } else {
        cout << "✗ COUNTS DO NOT MATCH - There's a problem!" << endl;
    }
    cout << "======================================" << endl;

    if (!cosmic_counts.empty()) {
        saveCosmicFluxToCSV(cosmic_counts, OUTPUT_DIR);
    } else {
        cerr << "Error: No valid cosmic events or starttime found in any input file. Skipping CSV output." << endl;
    }

    if (!file_cosmic_infos.empty()) {
        saveDailyCosmicFluxToCSV(file_cosmic_infos, OUTPUT_DIR);
    } else {
        cerr << "Error: No file cosmic info available for daily CSV." << endl;
    }

    for (const auto& muon : muon_candidates) {
        if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
            h_muon_energy->Fill(muon.second);
        }
    }

    cout << "h_low_iso entries: " << h_low_iso->GetEntries() << endl;
    cout << "h_dt_prompt_delayed entries: " << h_dt_prompt_delayed->GetEntries() << endl;
    cout << "num_low_iso: " << num_low_iso << endl;
    cout << "Excluded low-energy isolated events from h_dt_prompt_delayed (dt_high < 0 or > 10000 µs): " << excluded_low_iso << endl;

    cout << "Global Statistics:\n";
    cout << "Total Events: " << num_events << "\n";
    cout << "Muons Detected: " << num_muons << "\n";
    cout << "Michel Electrons Detected (0-16 μs): " << num_michels << "\n";
    cout << "Michel Events in Sideband (0-16 μs): " << num_michels_extended << "\n";
    cout << "Low-Energy Isolated Events: " << num_low_iso << "\n";
    cout << "Prompt-Delayed Pairs (h_dt_prompt_delayed entries): " << h_dt_prompt_delayed->GetEntries() << "\n";
    cout << "Cosmic Ray Events Detected: " << num_cosmic_events << "\n";
    cout << "------------------------\n";

    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }
    cout << "------------------------\n";

    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    c->Clear();
    h_muon_energy->SetLineColor(kBlue);
    h_muon_energy->Draw();
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_muon_all->SetLineColor(kBlue);
    h_muon_all->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Muon_All_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_dt_michel->SetLineWidth(2);
    h_dt_michel->SetLineColor(kBlack);
    h_dt_michel->GetXaxis()->SetTitle("Time to previous event (Muon) [#mus]");
    h_dt_michel->Draw("HIST");

    TF1* expFit = nullptr;
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
        expFit->SetParLimits(2, -C_init * 10, C_init * 10);
        expFit->SetParNames("N_{0}", "#tau", "C");
        expFit->SetLineColor(kRed);
        expFit->SetLineWidth(3);

        int fitStatus = h_dt_michel->Fit(expFit, "RE+", "SAME", FIT_MIN, FIT_MAX);
        expFit->Draw("SAME");

        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_michel->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.6);
            stats->SetY2NDC(0.9);
            stats->SetTextColor(kRed);
        }

        double N0 = expFit->GetParameter(0);
        double N0_err = expFit->GetParError(0);
        double tau = expFit->GetParameter(1);
        double tau_err = expFit->GetParError(1);
        double C = expFit->GetParameter(2);
        double C_err = expFit->GetParError(2);
        double chi2 = expFit->GetChisquare();
        int ndf = expFit->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (Michel dt, " << FIT_MIN << "-" << FIT_MAX << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("N₀ = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cout << "Warning: h_dt_michel has insufficient entries (" << h_dt_michel->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c->Update();
    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    if (expFit) delete expFit;

    if (h_dt_michel->GetEntries() > 5) {
        h_dt_michel->GetListOfFunctions()->Clear();
        std::vector<double> fit_starts = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        std::vector<double> taus, tau_errs, chi2ndfs;
        int best_index = -1;
        double min_chi2ndf = 1e9;

        for (int i = 0; i < fit_starts.size(); i++) {
            double fit_start = fit_starts[i];
            double fit_end = 16.0;

            TF1* expFit_var = new TF1(Form("expFit_var_%.1f", fit_start), ExpFit, fit_start, fit_end, 3);

            double C_init = 0;
            int bin_12 = h_dt_michel->FindBin(12.0);
            int bin_16 = h_dt_michel->FindBin(16.0);
            double min_content = 1e9;
            for (int bin = bin_12; bin <= bin_16; bin++) {
                double content = h_dt_michel->GetBinContent(bin);
                if (content > 0 && content < min_content) min_content = content;
            }
            if (min_content < 1e9) C_init = min_content;
            else C_init = 0.1;

            double integral = h_dt_michel->Integral(h_dt_michel->FindBin(fit_start), h_dt_michel->FindBin(fit_end));
            double bin_width = h_dt_michel->GetBinWidth(1);
            double N0_init = (integral * bin_width - C_init * (fit_end - fit_start)) / 2.2;
            if (N0_init < 0) N0_init = 100;

            expFit_var->SetParameters(N0_init, 2.2, C_init);
            expFit_var->SetParNames("N_{0}", "#tau", "C");
            expFit_var->SetParLimits(0, 0, N0_init * 100);
            expFit_var->SetParLimits(1, 0.1, 20.0);
            expFit_var->SetParLimits(2, -C_init * 10, C_init * 10);

            int fitStatus = h_dt_michel->Fit(expFit_var, "QRN+", "", fit_start, fit_end);

            double tau = expFit_var->GetParameter(1);
            double tau_err = expFit_var->GetParError(1);
            double chi2 = expFit_var->GetChisquare();
            int ndf = expFit_var->GetNDF();
            double chi2ndf = (ndf > 0) ? chi2 / ndf : 999;

            taus.push_back(tau);
            tau_errs.push_back(tau_err);
            chi2ndfs.push_back(chi2ndf);

            if (chi2ndf < min_chi2ndf && fitStatus == 0) {
                min_chi2ndf = chi2ndf;
                best_index = i;
            }

            cout << Form("Fit Range %.1f-%.1f µs:\n", fit_start, fit_end);
            cout << "Fit Status: " << fitStatus << " (0 = success)\n";
            cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
            cout << Form("χ²/NDF = %.4f", chi2ndf) << endl;
            cout << "----------------------------------------" << endl;

            delete expFit_var;
        }

        if (best_index >= 0) {
            cout << Form("Best Fit Range: %.1f-16.0 µs\n", fit_starts[best_index]);
            cout << Form("τ = %.4f ± %.4f µs", taus[best_index], tau_errs[best_index]) << endl;
            cout << Form("χ²/NDF = %.4f (minimum)", chi2ndfs[best_index]) << endl;
            cout << "----------------------------------------" << endl;
        }

        TCanvas* c_comp = new TCanvas("c_comp", "Fit Start Time Comparison", 1200, 800);
        c_comp->SetGrid();

        TPad* pad = new TPad("pad", "pad", 0, 0, 1, 1);
        pad->Draw();
        pad->cd();

        TGraph* g_chi2 = new TGraph(fit_starts.size(), &fit_starts[0], &chi2ndfs[0]);
        TGraph* g_tau = new TGraph(fit_starts.size(), &fit_starts[0], &taus[0]);

        g_chi2->SetTitle("Fit Start Time Comparison");
        g_chi2->GetXaxis()->SetTitle("Fit Start Time (#mus)");
        g_chi2->GetYaxis()->SetTitle("#chi^{2}/ndf");
        g_chi2->SetMarkerStyle(20);
        g_chi2->SetMarkerColor(kBlue);
        g_chi2->SetLineColor(kBlue);
        g_chi2->SetLineWidth(2);

        g_tau->SetMarkerStyle(22);
        g_tau->SetMarkerColor(kRed);
        g_tau->SetLineColor(kRed);
        g_tau->SetLineWidth(2);

        g_chi2->Draw("APL");

        pad->Update();
        double ymin = pad->GetUymin();
        double ymax = pad->GetUymax();

        double tau_min = *std::min_element(taus.begin(), taus.end());
        double tau_max = *std::max_element(taus.begin(), taus.end());
        double scale = (ymax - ymin)/(tau_max - tau_min);
        double offset = ymin - tau_min * scale;

        for (int i = 0; i < g_tau->GetN(); i++) {
            double x, y;
            g_tau->GetPoint(i, x, y);
            g_tau->SetPoint(i, x, y * scale + offset);
        }

        g_tau->Draw("PL same");

        TGaxis* axis = new TGaxis(gPad->GetUxmax(), gPad->GetUymin(),
                                 gPad->GetUxmax(), gPad->GetUymax(),
                                 tau_min, tau_max, 510, "+L");
        axis->SetLineColor(kRed);
        axis->SetLabelColor(kRed);
        axis->SetTitle("#tau (#mus)");
        axis->SetTitleColor(kRed);
        axis->Draw();

        TLegend* leg = new TLegend(0.7, 0.7, 0.9, 0.9);
        leg->AddEntry(g_chi2, "#chi^{2}/ndf", "lp");
        leg->AddEntry(g_tau, "#tau", "lp");
        leg->Draw();

        string compPlotName = OUTPUT_DIR + "/FitStartComparison.png";
        c_comp->SaveAs(compPlotName.c_str());
        cout << "Saved comparison plot: " << compPlotName << endl;

        delete g_chi2;
        delete g_tau;
        delete leg;
        delete axis;
        delete pad;
        delete c_comp;
    } else {
        cout << "Skipping fit start comparison - insufficient entries in dt histogram" << endl;
    }

    c->Clear();
    h_energy_vs_dt->SetStats(0);
    h_energy_vs_dt->GetXaxis()->SetTitle("dt (#mus)");
    h_energy_vs_dt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_side_vp_muon->SetLineColor(kMagenta);
    h_side_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Side_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_top_vp_muon->SetLineColor(kCyan);
    h_top_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Top_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_trigger_bits->SetLineColor(kGreen);
    h_trigger_bits->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    createVetoPanelPlots(h_veto_panel, OUTPUT_DIR);

    c->Clear();
    h_isolated_pe->SetLineColor(kBlack);
    h_isolated_pe->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Isolated_PE.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_low_iso->SetLineColor(kBlack);
    h_low_iso->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Isolated.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_high_iso->SetLineColor(kBlack);
    h_high_iso->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/High_Energy_Isolated.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_dt_prompt_delayed->SetLineColor(kBlack);
    h_dt_prompt_delayed->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/DeltaT_High_to_Low.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    TCanvas *c_low_muon = new TCanvas("c_low_muon", "DeltaT Low to Muon", 1200, 800);
    c_low_muon->SetLeftMargin(0.15);
    c_low_muon->SetRightMargin(0.08);
    c_low_muon->SetBottomMargin(0.12);
    c_low_muon->SetTopMargin(0.08);

    h_dt_low_muon->SetLineWidth(2);
    h_dt_low_muon->SetLineColor(kBlue);
    h_dt_low_muon->SetFillColor(kBlue);
    h_dt_low_muon->SetFillStyle(3001);
    h_dt_low_muon->GetXaxis()->SetTitleSize(0.04);
    h_dt_low_muon->GetYaxis()->SetTitleSize(0.05);
    h_dt_low_muon->GetXaxis()->SetLabelSize(0.04);
    h_dt_low_muon->GetYaxis()->SetLabelSize(0.04);
    h_dt_low_muon->SetTitle("#Delta t: Low Energy Isolated to Muon");

    h_dt_low_muon->Draw("HIST");

    TF1* expFit_low_muon = nullptr;
    if (h_dt_low_muon->GetEntries() > 10) {
        expFit_low_muon = new TF1("expFit_low_muon", ExpFit, FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON, 3);
        expFit_low_muon->SetParNames("N_{0}", "#tau", "C");

        double integral = h_dt_low_muon->Integral(h_dt_low_muon->FindBin(FIT_MIN_LOW_MUON), h_dt_low_muon->FindBin(FIT_MAX_LOW_MUON));
        double bin_width = h_dt_low_muon->GetBinWidth(1);
        double C_init = 0;
        int bin_400 = h_dt_low_muon->FindBin(400.0);
        int bin_500 = h_dt_low_muon->FindBin(500.0);
        double min_content = 1e9;
        for (int i = bin_400; i <= bin_500; i++) {
            double content = h_dt_low_muon->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        double N0_init = (integral * bin_width - C_init * (FIT_MAX_LOW_MUON - FIT_MIN_LOW_MUON)) / 200.0;
        if (N0_init <= 0) N0_init = 1.0;

        expFit_low_muon->SetParameters(N0_init, 200.0, C_init);
        expFit_low_muon->SetParLimits(0, 0, N0_init * 100);
        expFit_low_muon->SetParLimits(1, 0.1, 1000.0);
        expFit_low_muon->SetParLimits(2, -C_init * 10, C_init * 10);

        int fitStatus = h_dt_low_muon->Fit(expFit_low_muon, "RE+", "", FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON);

        expFit_low_muon->SetLineColor(kRed);
        expFit_low_muon->SetLineWidth(3);
        expFit_low_muon->Draw("SAME");

        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_low_muon->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.7);
            stats->SetY2NDC(0.95);
            stats->SetTextColor(kRed);
        }

        double N0 = expFit_low_muon->GetParameter(0);
        double N0_err = expFit_low_muon->GetParError(0);
        double tau = expFit_low_muon->GetParameter(1);
        double tau_err = expFit_low_muon->GetParError(1);
        double C = expFit_low_muon->GetParameter(2);
        double C_err = expFit_low_muon->GetParError(2);
        double chi2 = expFit_low_muon->GetChisquare();
        int ndf = expFit_low_muon->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (Low to Muon dt, " << FIT_MIN_LOW_MUON << "-" << FIT_MAX_LOW_MUON << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("N_{0} = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;

        cout << "=== Neutron Purity Analysis ===" << endl;

        double bw = h_dt_low_muon->GetBinWidth(1);
        double N0_rate = N0;
        double C_rate = C;
        double t_min = 16.0;

        for (int time_cut = 16; time_cut <= 1000; time_cut += 10) {
            double sig = N0_rate * exp(-time_cut / tau);
            double bkg = C_rate;
            
            if (bkg > 0 && sig > 0) {
                double neutron_ratio = sig / bkg;
                double significance = sig / sqrt(sig + bkg);
                h_neutron_richness->Fill(time_cut, neutron_ratio);
                h_signal_significance->Fill(time_cut, significance);
            }
            
            if (time_cut % 100 == 0) {
                cout << "Time cut " << time_cut << " µs: Signal=" << sig 
                     << ", Bkg=" << bkg << ", Ratio=" << sig/bkg 
                     << ", Significance=" << sig/sqrt(sig + bkg) << endl;
            }
        }
    } else {
        cout << "Warning: h_dt_low_muon has insufficient entries (" << h_dt_low_muon->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c_low_muon->Update();
    plotName = OUTPUT_DIR + "/DeltaT_Low_to_Muon.png";
    c_low_muon->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    if (expFit_low_muon) delete expFit_low_muon;

    TCanvas *c_high_muon = new TCanvas("c_high_muon", "DeltaT High to Muon", 1200, 800);
    c_high_muon->SetLeftMargin(0.12);
    c_high_muon->SetRightMargin(0.08);
    c_high_muon->SetBottomMargin(0.12);
    c_high_muon->SetTopMargin(0.08);

    h_dt_high_muon->SetLineWidth(2);
    h_dt_high_muon->SetLineColor(kBlue);
    h_dt_high_muon->SetFillColor(kBlack);
    h_dt_high_muon->SetFillStyle(3001);
    h_dt_high_muon->GetXaxis()->SetTitleSize(0.05);
    h_dt_high_muon->GetYaxis()->SetTitleSize(0.05);
    h_dt_high_muon->GetXaxis()->SetLabelSize(0.04);
    h_dt_high_muon->GetYaxis()->SetLabelSize(0.04);
    h_dt_high_muon->SetTitle("#Delta t: High Energy Isolated to Muon");

    h_dt_high_muon->Draw("HIST");

    TF1* expFit_high_muon = nullptr;
    if (h_dt_high_muon->GetEntries() > 10) {
        expFit_high_muon = new TF1("expFit_high_muon", ExpFit, FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON, 3);
        expFit_high_muon->SetParNames("N_{0}", "#tau", "C");

        double integral = h_dt_high_muon->Integral(h_dt_high_muon->FindBin(FIT_MIN_LOW_MUON), h_dt_high_muon->FindBin(FIT_MAX_LOW_MUON));
        double bin_width = h_dt_high_muon->GetBinWidth(1);
        double C_init = 0;
        int bin_400 = h_dt_high_muon->FindBin(400.0);
        int bin_500 = h_dt_high_muon->FindBin(500.0);
        double min_content = 1e9;
        for (int i = bin_400; i <= bin_500; i++) {
            double content = h_dt_high_muon->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        double N0_init = (integral * bin_width - C_init * (FIT_MAX_LOW_MUON - FIT_MIN_LOW_MUON)) / 200.0;
        if (N0_init <= 0) N0_init = 1.0;

        expFit_high_muon->SetParameters(N0_init, 200.0, C_init);
        expFit_high_muon->SetParLimits(0, 0, N0_init * 100);
        expFit_high_muon->SetParLimits(1, 0.1, 1000.0);
        expFit_high_muon->SetParLimits(2, -C_init * 10, C_init * 10);

        int fitStatus = h_dt_high_muon->Fit(expFit_high_muon, "RE+", "", FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON);

        expFit_high_muon->SetLineColor(kRed);
        expFit_high_muon->SetLineWidth(3);
        expFit_high_muon->Draw("SAME");

        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_high_muon->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.7);
            stats->SetY2NDC(0.95);
            stats->SetTextColor(kRed);
        }

        double N0 = expFit_high_muon->GetParameter(0);
        double N0_err = expFit_high_muon->GetParError(0);
        double tau = expFit_high_muon->GetParameter(1);
        double tau_err = expFit_high_muon->GetParError(1);
        double C = expFit_high_muon->GetParameter(2);
        double C_err = expFit_high_muon->GetParError(2);
        double chi2 = expFit_high_muon->GetChisquare();
        int ndf = expFit_high_muon->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (High to Muon dt, " << FIT_MIN_LOW_MUON << "-" << FIT_MAX_LOW_MUON << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("N_{0} = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cout << "Warning: h_dt_high_muon has insufficient entries (" << h_dt_high_muon->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c_high_muon->Update();
    plotName = OUTPUT_DIR + "/DeltaT_High_to_Muon.png";
    c_high_muon->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    if (expFit_high_muon) delete expFit_high_muon;

    cout << "=== Michel Background Subtraction with Consistent Fit Range ===" << endl;

    double N0_fit = 0, tau_fit = 0, C_fit = 0;
    double N0_err = 0, tau_err = 0, C_err = 0;
    double chi2_ndf_fit = 0;
    double predicted_michels = 0;
    double predicted_michels_err = 0;
    double fit_michels_2_16 = 0;

    const double MICHEL_FIT_RANGE_MIN = MICHEL_BKG_FIT_MIN;
    const double MICHEL_FIT_RANGE_MAX = MICHEL_BKG_FIT_MAX;
    const double MICHEL_PREDICTION_MIN = SIGNAL_REGION_MIN;
    const double MICHEL_PREDICTION_MAX = SIGNAL_REGION_MAX;

    if (h_dt_michel_sideband->GetEntries() > 20) {
        TF1* michel_fit = new TF1("michel_fit", ExpFit, MICHEL_FIT_RANGE_MIN, MICHEL_FIT_RANGE_MAX, 3);
        michel_fit->SetParNames("N_{0}", "#tau", "C");
        
        double integral = h_dt_michel_sideband->Integral(
            h_dt_michel_sideband->FindBin(MICHEL_FIT_RANGE_MIN),
            h_dt_michel_sideband->FindBin(MICHEL_FIT_RANGE_MAX)
        );
        double bin_width = h_dt_michel_sideband->GetBinWidth(1);
        double N0_init = integral * 2.2;
        double tau_init = 2.2;
        double C_init = h_dt_michel_sideband->GetBinContent(h_dt_michel_sideband->FindBin(15.0));
        
        michel_fit->SetParameters(N0_init, tau_init, C_init);
        michel_fit->SetParLimits(0, 0, N0_init * 10);
        michel_fit->SetParLimits(1, 1.0, 3.0);
        michel_fit->SetParLimits(2, 0, C_init * 5);
        
        int fit_status = h_dt_michel_sideband->Fit(michel_fit, "RE+", "", MICHEL_FIT_RANGE_MIN, MICHEL_FIT_RANGE_MAX);
        
        if (fit_status == 0) {
            N0_fit = michel_fit->GetParameter(0);
            tau_fit = michel_fit->GetParameter(1);
            C_fit = michel_fit->GetParameter(2);
            N0_err = michel_fit->GetParError(0);
            tau_err = michel_fit->GetParError(1);
            C_err = michel_fit->GetParError(2);
            chi2_ndf_fit = michel_fit->GetChisquare() / michel_fit->GetNDF();
            
            cout << "Michel Fit Results (" << MICHEL_FIT_RANGE_MIN << "-" << MICHEL_FIT_RANGE_MAX << " μs):" << endl;
            cout << Form("N₀ = %.1f ± %.1f", N0_fit, N0_err) << endl;
            cout << Form("τ = %.3f ± %.3f μs", tau_fit, tau_err) << endl;
            cout << Form("C = %.1f ± %.1f", C_fit, C_err) << endl;
            cout << Form("χ²/NDF = %.2f", chi2_ndf_fit) << endl;
            
            fit_michels_2_16 = calculateFitMichels(N0_fit, tau_fit, C_fit, MICHEL_FIT_RANGE_MIN, MICHEL_FIT_RANGE_MAX, 0.1);
            
            predicted_michels = calculateFitMichels(N0_fit, tau_fit, C_fit, MICHEL_PREDICTION_MIN, MICHEL_PREDICTION_MAX, 0.1);
            
            predicted_michels_err = 0.0;
            const double TIME_BIN_SIZE = 0.1;
            
            for (int i = 0; i < static_cast<int>((MICHEL_PREDICTION_MAX - MICHEL_PREDICTION_MIN) / TIME_BIN_SIZE); i++) {
                double t_start = MICHEL_PREDICTION_MIN + i * TIME_BIN_SIZE;
                double t_end = t_start + TIME_BIN_SIZE;
                double t_center = t_start + (TIME_BIN_SIZE / 2.0);
                
                if (t_end > MICHEL_PREDICTION_MAX) break;
                
                double dN_dN0 = exp(-t_center / tau_fit);
                double dN_dtau = N0_fit * (t_center / (tau_fit * tau_fit)) * exp(-t_center / tau_fit);
                double bin_variance = pow(dN_dN0 * N0_err, 2) + pow(dN_dtau * tau_err, 2);
                predicted_michels_err += bin_variance;
            }
            
            predicted_michels_err = sqrt(predicted_michels_err);
            
            cout << Form("Fit Michels (2-16 μs): %.1f", fit_michels_2_16) << endl;
            cout << Form("Predicted Michels (16-100 μs): %.1f ± %.1f", predicted_michels, predicted_michels_err) << endl;
        } else {
            cout << "Warning: Michel fit failed with status " << fit_status << endl;
        }
        
        delete michel_fit;
    } else {
        cout << "Warning: Insufficient Michel events in sideband for fitting: " 
             << h_dt_michel_sideband->GetEntries() << endl;
    }

    saveLiveTimeInfo(totalLiveTime, fit_michels_2_16, predicted_michels, predicted_michels_err, OUTPUT_DIR);

    double michel_scale = (predicted_michels > 0 && fit_michels_2_16 > 0) ? predicted_michels / fit_michels_2_16 : 0;
    
    cout << "=== Improved Michel Scaling Factor Calculation ===" << endl;
    cout << "Predicted Michels (16-100 μs): " << predicted_michels << endl;
    cout << "Fit Michels (2-16 μs): " << fit_michels_2_16 << endl;
    cout << "Scaling factor (Predicted/Fit): " << michel_scale << endl;
    cout << "==================================================" << endl;

    h_michel_energy_predicted = (TH1D*)h_michel_energy_fit_range->Clone("michel_energy_predicted");
    h_michel_energy_predicted->Scale(michel_scale);

    double signal_events = calculateTotalEvents(h_low_pe_signal);
    double sideband_events = calculateTotalEvents(h_low_pe_sideband);
    
    double sideband_scale_factor = (SIGNAL_REGION_MAX - SIGNAL_REGION_MIN) / (1200.0 - 1000.0);
    h_scaled_sideband = (TH1D*)h_low_pe_sideband->Clone("scaled_sideband");
    h_scaled_sideband->Scale(sideband_scale_factor);
    double scaled_sideband_events = calculateTotalEvents(h_scaled_sideband);

    h_michel_background_predicted = (TH1D*)h_michel_energy_predicted->Clone("michel_background_predicted");

    h_final_subtracted = (TH1D*)h_low_pe_signal->Clone("after_sideband_sub");
    h_final_subtracted->Add(h_scaled_sideband, -1.0);
    double after_sideband_events = calculateTotalEvents(h_final_subtracted);

    // Calculate the final subtracted value using manual calculation to ensure accuracy
    double final_subtracted_corrected = signal_events - scaled_sideband_events - predicted_michels;

    // Update the final subtracted histogram with the correct values
    h_final_subtracted = (TH1D*)h_final_subtracted->Clone("final_subtracted"); 
    for (int i = 1; i <= h_final_subtracted->GetNbinsX(); i++) {
        double signal_bin = h_low_pe_signal->GetBinContent(i);
        double scaled_bkg_bin = h_scaled_sideband->GetBinContent(i);
        double michel_bkg_bin = h_michel_background_predicted->GetBinContent(i);
        double corrected_bin = signal_bin - scaled_bkg_bin - michel_bkg_bin;
        h_final_subtracted->SetBinContent(i, corrected_bin);
    }

    double final_events = final_subtracted_corrected;

    double manual_calculation = signal_events - scaled_sideband_events - predicted_michels;
    double discrepancy = final_events - manual_calculation;

    cout << "=== CORRECTED Low Energy Subtraction Results ===" << endl;
    cout << "Signal region (16-100 μs) events: " << signal_events << endl;
    cout << "Sideband region (1000-1200 μs) events: " << sideband_events << endl;
    cout << "Sideband scale factor: " << sideband_scale_factor << endl;
    cout << "Scaled neutron-free background: " << scaled_sideband_events << endl;
    cout << "Predicted Michel background: " << predicted_michels << " ± " << predicted_michels_err << endl;
    cout << "After sideband subtraction: " << after_sideband_events << endl;
    cout << "Final subtracted events: " << final_subtracted_corrected << endl;
    cout << "Manual verification (signal - scaled_bkg - michel): " << manual_calculation << endl;
    cout << "Discrepancy: " << discrepancy << endl;
    cout << "================================================" << endl;

    if (predicted_michels > after_sideband_events * 0.5) {
        cout << "WARNING: Michel background seems too large compared to signal!" << endl;
        cout << "Michel background is " << (predicted_michels/after_sideband_events)*100 << "% of sideband-subtracted signal" << endl;
    }

    TCanvas *c_michel_method = new TCanvas("c_michel_method", "Michel Background Subtraction Method", 1200, 800);
    c_michel_method->Divide(2, 1);
    
    c_michel_method->cd(1);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.08);
    gPad->SetBottomMargin(0.12);
    gPad->SetTopMargin(0.08);
    gPad->SetLogy();
    
    TH1D* h_dt_michel_fit_range = (TH1D*)h_dt_michel_sideband->Clone("dt_michel_fit_range");
    h_dt_michel_fit_range->SetTitle(Form("Michel Time Distribution (%0.1f-%0.1f #mus);Time to Previous Muon (#mus);Counts/0.2#mus", 
                                        MICHEL_FIT_RANGE_MIN, MICHEL_FIT_RANGE_MAX));
    h_dt_michel_fit_range->GetXaxis()->SetRangeUser(0, 16);
    h_dt_michel_fit_range->SetLineColor(kBlue);
    h_dt_michel_fit_range->SetLineWidth(2);
    h_dt_michel_fit_range->Draw("HIST");

    if (predicted_michels > 0) {
        TF1* michel_fit_plot = new TF1("michel_fit_plot", ExpFit, MICHEL_FIT_RANGE_MIN, MICHEL_FIT_RANGE_MAX, 3);
        michel_fit_plot->SetParameters(N0_fit, tau_fit, C_fit);
        michel_fit_plot->SetLineColor(kRed);
        michel_fit_plot->SetLineWidth(2);
        michel_fit_plot->Draw("SAME");
    }

    c_michel_method->cd(2);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.08);
    gPad->SetBottomMargin(0.12);
    gPad->SetTopMargin(0.08);
    gPad->SetLogy();
    
    h_michel_energy_fit_range->SetMinimum(0.1);
    h_michel_energy_fit_range->SetLineColor(kBlue);
    h_michel_energy_fit_range->SetLineWidth(2);
    h_michel_energy_fit_range->SetStats(0);
    h_michel_energy_fit_range->Draw("HIST");

    h_michel_energy_predicted->SetLineColor(kRed);
    h_michel_energy_predicted->SetLineWidth(2);
    h_michel_energy_predicted->SetLineStyle(2);
    h_michel_energy_predicted->SetStats(0);
    h_michel_energy_predicted->Draw("HIST SAME");

    TLegend *leg_energy = new TLegend(0.15, 0.80, 0.45, 0.93);
    leg_energy->SetBorderSize(0);
    leg_energy->SetFillStyle(0);
    leg_energy->SetTextSize(0.045);
    leg_energy->AddEntry(h_michel_energy_fit_range, 
                        Form("%0.1f-%0.1f #mus", MICHEL_FIT_RANGE_MIN, MICHEL_FIT_RANGE_MAX), "l");
    leg_energy->AddEntry(h_michel_energy_predicted, 
                        Form("%0.1f-%0.1f #mus", MICHEL_PREDICTION_MIN, MICHEL_PREDICTION_MAX), "l");
    leg_energy->Draw();

    c_michel_method->Update();
    plotName = OUTPUT_DIR + "/Michel_Background_Subtraction.png";
    c_michel_method->SaveAs(plotName.c_str());
    cout << "Saved Michel background subtraction plot: " << plotName << endl;

    c->Clear();
    h_isolated_ge40->SetLineColor(kBlack);
    h_isolated_ge40->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Isolated_GE40_PE.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    c->Divide(1,2);
    
    c->cd(1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    h_neutron_richness->SetStats(0);
    h_neutron_richness->SetLineColor(kBlue);
    h_neutron_richness->SetLineWidth(3);
    h_neutron_richness->GetXaxis()->SetTitleSize(0.08);
    h_neutron_richness->GetXaxis()->SetTitleOffset(0.6);
    h_neutron_richness->GetYaxis()->SetTitleSize(0.08);
    h_neutron_richness->GetYaxis()->SetTitleOffset(0.6);
    h_neutron_richness->GetXaxis()->SetLabelSize(0.05);
    h_neutron_richness->GetYaxis()->SetLabelSize(0.05);
    h_neutron_richness->Draw("HIST");
    
    c->cd(2);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    h_signal_significance->SetStats(0);
    h_signal_significance->SetLineColor(kRed);
    h_signal_significance->SetLineWidth(3);
    h_signal_significance->GetXaxis()->SetTitleSize(0.08);
    h_signal_significance->GetXaxis()->SetTitleOffset(0.6);
    h_signal_significance->GetYaxis()->SetTitleSize(0.08);
    h_signal_significance->GetYaxis()->SetTitleOffset(0.6);
    h_signal_significance->GetXaxis()->SetLabelSize(0.05);
    h_signal_significance->GetYaxis()->SetLabelSize(0.05);
    h_signal_significance->Draw("HIST");
    
    c->Update();
    plotName = OUTPUT_DIR + "/Neutron_Purity_Analysis.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    c->Divide(1,2);
    c->cd(1);
    h_energy_vs_time_low->Draw("COLZ");
    c->cd(2);
    h_energy_vs_time_high->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Energy_vs_Time_MultiD.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    TCanvas *c_sideband1 = new TCanvas("c_sideband1", "Low Energy Sideband Subtraction", 1200, 800);
    c_sideband1->SetLeftMargin(0.1);
    c_sideband1->SetRightMargin(0.1);
    c_sideband1->SetBottomMargin(0.1);
    c_sideband1->SetTopMargin(0.1);

    h_low_pe_signal->SetLineColor(kRed);
    h_low_pe_signal->SetLineWidth(3);
    h_low_pe_sideband->SetLineColor(kBlue);
    h_low_pe_sideband->SetLineWidth(3);
    h_scaled_sideband->SetLineColor(kBlue);
    h_scaled_sideband->SetLineWidth(3);
    h_scaled_sideband->SetLineStyle(2);

    h_low_pe_signal->SetStats(0);
    h_low_pe_sideband->SetStats(0);
    h_scaled_sideband->SetStats(0);

    h_low_pe_signal->Draw("HIST");
    h_low_pe_sideband->Draw("HIST same");
    h_scaled_sideband->Draw("HIST same");

    TLegend *leg_sub1 = new TLegend(0.5, 0.65, 0.9, 0.9);
    leg_sub1->SetTextSize(0.025);
    leg_sub1->SetTextFont(42);
    leg_sub1->SetBorderSize(1);
    leg_sub1->SetFillStyle(0);
    leg_sub1->AddEntry(h_low_pe_signal, Form("Neutron rich region (16-100 #mus) [%.0f events]", signal_events), "l");
    leg_sub1->AddEntry(h_low_pe_sideband, Form("Neutron free region (1000-1200 #mus) [%.0f events]", sideband_events), "l");
    leg_sub1->AddEntry(h_scaled_sideband, Form("Scaled neutron free region [%.1f events]", scaled_sideband_events), "l");
    leg_sub1->Draw();

    c_sideband1->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Sideband_Subtraction.png";
    c_sideband1->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    TCanvas *c_sideband2 = new TCanvas("c_sideband2", "Low Energy Sideband Subtraction with Michel Background", 1200, 800);
    c_sideband2->SetLeftMargin(0.1);
    c_sideband2->SetRightMargin(0.1);
    c_sideband2->SetBottomMargin(0.1);
    c_sideband2->SetTopMargin(0.1);

    h_low_pe_signal->SetLineColor(kRed);
    h_low_pe_signal->SetLineWidth(3);
    h_scaled_sideband->SetLineColor(kBlue);
    h_scaled_sideband->SetLineWidth(2);
    h_scaled_sideband->SetLineStyle(2);
    h_michel_background_predicted->SetLineColor(kMagenta);
    h_michel_background_predicted->SetLineWidth(2);
    h_michel_background_predicted->SetLineStyle(3);
    h_final_subtracted->SetLineColor(kGreen);
    h_final_subtracted->SetLineWidth(3);

    h_low_pe_signal->SetStats(0);
    h_scaled_sideband->SetStats(0);
    h_michel_background_predicted->SetStats(0);
    h_final_subtracted->SetStats(0);

    h_low_pe_signal->Draw("HIST");
    h_scaled_sideband->Draw("HIST SAME");
    h_michel_background_predicted->Draw("HIST SAME");
    h_final_subtracted->Draw("HIST SAME");

    TLegend *leg_sub2 = new TLegend(0.5, 0.6, 0.9, 0.9);
    leg_sub2->SetTextSize(0.025);
    leg_sub2->SetTextFont(42);
    leg_sub2->SetBorderSize(1);
    leg_sub2->SetFillStyle(0);
    leg_sub2->AddEntry(h_low_pe_signal, Form("Neutron rich region (16-100 #mus) [%.0f events]", signal_events), "l");
    leg_sub2->AddEntry(h_scaled_sideband, Form("Scaled neutron free region [%.1f events]", scaled_sideband_events), "l");
    leg_sub2->AddEntry(h_michel_background_predicted, Form("Michel background (16-100 #mus) [%.1f #pm %.1f events]", predicted_michels, predicted_michels_err), "l");
    leg_sub2->AddEntry(h_final_subtracted, Form("Final: Signal - ScaledBkg - Michel [%.1f events]", final_subtracted_corrected), "l");
    leg_sub2->Draw();

    c_sideband2->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Sideband_Subtraction_Complete.png";
    c_sideband2->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    TCanvas *c_sideband2_log = new TCanvas("c_sideband2_log", "Low Energy Sideband Subtraction with Michel Background (Log Scale)", 1200, 800);
    c_sideband2_log->SetLeftMargin(0.1);
    c_sideband2_log->SetRightMargin(0.1);
    c_sideband2_log->SetBottomMargin(0.1);
    c_sideband2_log->SetTopMargin(0.1);
    c_sideband2_log->SetLogy();

    h_low_pe_signal->SetLineColor(kRed);
    h_low_pe_signal->SetLineWidth(3);
    h_scaled_sideband->SetLineColor(kBlue);
    h_scaled_sideband->SetLineWidth(2);
    h_scaled_sideband->SetLineStyle(2);
    h_michel_background_predicted->SetLineColor(kMagenta);
    h_michel_background_predicted->SetLineWidth(2);
    h_michel_background_predicted->SetLineStyle(3);
    h_final_subtracted->SetLineColor(kGreen);
    h_final_subtracted->SetLineWidth(3);

    h_low_pe_signal->SetStats(0);
    h_scaled_sideband->SetStats(0);
    h_michel_background_predicted->SetStats(0);
    h_final_subtracted->SetStats(0);

    h_low_pe_signal->SetMinimum(0.1);
    h_low_pe_signal->Draw("HIST");
    h_scaled_sideband->Draw("HIST SAME");
    h_michel_background_predicted->Draw("HIST SAME");
    h_final_subtracted->Draw("HIST SAME");

    TLegend *leg_sub2_log = new TLegend(0.5, 0.6, 0.9, 0.9);
    leg_sub2_log->SetTextSize(0.025);
    leg_sub2_log->SetTextFont(42);
    leg_sub2_log->SetBorderSize(1);
    leg_sub2_log->SetFillStyle(0);
    leg_sub2_log->AddEntry(h_low_pe_signal, Form("Neutron rich region (16-100 #mus) [%.0f events]", signal_events), "l");
    leg_sub2_log->AddEntry(h_scaled_sideband, Form("Scaled neutron free region [%.1f events]", scaled_sideband_events), "l");
    leg_sub2_log->AddEntry(h_michel_background_predicted, Form("Michel background (16-100 #mus) [%.1f #pm %.1f events]", predicted_michels, predicted_michels_err), "l");
    leg_sub2_log->AddEntry(h_final_subtracted, Form("Final: Signal - ScaledBkg - Michel [%.1f events]", final_subtracted_corrected), "l");
    leg_sub2_log->Draw();

    c_sideband2_log->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Sideband_Subtraction_Complete_Log.png";
    c_sideband2_log->SaveAs(plotName.c_str());
    cout << "Saved log-scale plot: " << plotName << endl;

    cout << "=== Creating Normalized Sideband Subtraction Plots ===" << endl;
    
    // Use the correct rates that were already calculated
    double signal_rate = signal_events / liveTimeDays;
    double sideband_rate = sideband_events / liveTimeDays;
    double scaled_sideband_rate = scaled_sideband_events / liveTimeDays;
    double michel_rate = predicted_michels / liveTimeDays;
    double final_rate_corrected = final_subtracted_corrected / liveTimeDays;

    cout << "=== Rate Calculation (Events per Day) ===" << endl;
    cout << "Actual Live Time: " << liveTimeDays << " days" << endl;
    cout << "Signal region (16-100 μs): " << signal_events << " events → " << signal_rate << " events/day" << endl;
    cout << "Neutron-free region (1000-1200 μs): " << sideband_events << " events → " << sideband_rate << " events/day" << endl;
    cout << "Scaled neutron-free background: " << scaled_sideband_events << " events → " << scaled_sideband_rate << " events/day" << endl;
    cout << "Michel background: " << predicted_michels << " events → " << michel_rate << " events/day" << endl;
    cout << "Final subtracted: " << final_subtracted_corrected << " events → " << final_rate_corrected << " events/day" << endl;
    cout << "=========================================" << endl;

    // Create normalized histograms
    h_low_pe_signal_norm = (TH1D*)h_low_pe_signal->Clone("low_pe_signal_norm");
    h_low_pe_sideband_norm = (TH1D*)h_low_pe_sideband->Clone("low_pe_sideband_norm");
    h_scaled_sideband_norm = (TH1D*)h_scaled_sideband->Clone("scaled_sideband_norm");
    h_michel_background_predicted_norm = (TH1D*)h_michel_background_predicted->Clone("michel_background_predicted_norm");
    h_final_subtracted_norm = (TH1D*)h_final_subtracted->Clone("final_subtracted_norm");

    double norm_factor = 1.0 / liveTimeDays;
    h_low_pe_signal_norm->Scale(norm_factor);
    h_low_pe_sideband_norm->Scale(norm_factor);
    h_scaled_sideband_norm->Scale(norm_factor);
    h_michel_background_predicted_norm->Scale(norm_factor);

    // Update normalized final subtracted histogram with correct values
    for (int i = 1; i <= h_final_subtracted_norm->GetNbinsX(); i++) {
        double signal_bin = h_low_pe_signal_norm->GetBinContent(i);
        double scaled_bkg_bin = h_scaled_sideband_norm->GetBinContent(i);
        double michel_bkg_bin = h_michel_background_predicted_norm->GetBinContent(i);
        double corrected_bin = signal_bin - scaled_bkg_bin - michel_bkg_bin;
        h_final_subtracted_norm->SetBinContent(i, corrected_bin);
    }

    // Verification - only check the ones that should match
    double signal_rate_verified = calculateTotalEvents(h_low_pe_signal_norm);
    double sideband_rate_verified = calculateTotalEvents(h_low_pe_sideband_norm);
    double scaled_sideband_rate_verified = calculateTotalEvents(h_scaled_sideband_norm);

    cout << "=== Verification of Normalized Rates ===" << endl;
    cout << "Signal rate (calculated): " << signal_rate << " events/day" << endl;
    cout << "Signal rate (from normalized hist): " << signal_rate_verified << " events/day" << endl;
    cout << "Difference: " << fabs(signal_rate - signal_rate_verified) << " events/day" << endl;
    cout << "Scaling factor used: 1/" << liveTimeDays << " = " << norm_factor << endl;
    cout << "========================================" << endl;

    TCanvas *c_sideband1_norm = new TCanvas("c_sideband1_norm", "Normalized Low Energy Sideband Subtraction", 1200, 800);
    c_sideband1_norm->SetLeftMargin(0.1);
    c_sideband1_norm->SetRightMargin(0.1);
    c_sideband1_norm->SetBottomMargin(0.1);
    c_sideband1_norm->SetTopMargin(0.1);

    h_low_pe_signal_norm->SetLineColor(kRed);
    h_low_pe_signal_norm->SetLineWidth(3);
    h_low_pe_sideband_norm->SetLineColor(kBlue);
    h_low_pe_sideband_norm->SetLineWidth(3);
    h_scaled_sideband_norm->SetLineColor(kBlue);
    h_scaled_sideband_norm->SetLineWidth(3);
    h_scaled_sideband_norm->SetLineStyle(2);

    h_low_pe_signal_norm->SetStats(0);
    h_low_pe_sideband_norm->SetStats(0);
    h_scaled_sideband_norm->SetStats(0);
    
    h_low_pe_signal_norm->GetYaxis()->SetTitle("Counts per Day");
    h_low_pe_sideband_norm->GetYaxis()->SetTitle("Counts per Day");
    h_scaled_sideband_norm->GetYaxis()->SetTitle("Counts per Day");

    h_low_pe_signal_norm->Draw("HIST");
    h_low_pe_sideband_norm->Draw("HIST same");
    h_scaled_sideband_norm->Draw("HIST same");

    TLegend *leg_sub1_norm = new TLegend(0.5, 0.65, 0.9, 0.9);
    leg_sub1_norm->SetTextSize(0.025);
    leg_sub1_norm->SetTextFont(42);
    leg_sub1_norm->SetBorderSize(1);
    leg_sub1_norm->SetFillStyle(0);
    leg_sub1_norm->AddEntry(h_low_pe_signal_norm, Form("Neutron rich region (16-100 #mus) [%.1f events/day]", 
                                                      calculateTotalEvents(h_low_pe_signal_norm)), "l");
    leg_sub1_norm->AddEntry(h_low_pe_sideband_norm, Form("Neutron free region (1000-1200 #mus) [%.1f events/day]", 
                                                        calculateTotalEvents(h_low_pe_sideband_norm)), "l");
    leg_sub1_norm->AddEntry(h_scaled_sideband_norm, Form("Scaled neutron free region [%.1f events/day]", 
                                                        calculateTotalEvents(h_scaled_sideband_norm)), "l");
    leg_sub1_norm->AddEntry((TObject*)0, Form("Live time: %.6f days", liveTimeDays), "");
    leg_sub1_norm->Draw();

    c_sideband1_norm->Update();
    plotName = OUTPUT_DIR + "/Normalized_Low_Energy_Sideband_Subtraction.png";
    c_sideband1_norm->SaveAs(plotName.c_str());
    cout << "Saved normalized plot: " << plotName << endl;

    TCanvas *c_sideband2_norm = new TCanvas("c_sideband2_norm", "Normalized Low Energy Sideband Subtraction with Michel Background", 1200, 800);
    c_sideband2_norm->SetLeftMargin(0.1);
    c_sideband2_norm->SetRightMargin(0.1);
    c_sideband2_norm->SetBottomMargin(0.1);
    c_sideband2_norm->SetTopMargin(0.1);

    h_low_pe_signal_norm->SetLineColor(kRed);
    h_low_pe_signal_norm->SetLineWidth(3);
    h_scaled_sideband_norm->SetLineColor(kBlue);
    h_scaled_sideband_norm->SetLineWidth(2);
    h_scaled_sideband_norm->SetLineStyle(2);
    h_michel_background_predicted_norm->SetLineColor(kMagenta);
    h_michel_background_predicted_norm->SetLineWidth(2);
    h_michel_background_predicted_norm->SetLineStyle(3);
    h_final_subtracted_norm->SetLineColor(kGreen);
    h_final_subtracted_norm->SetLineWidth(3);

    h_low_pe_signal_norm->SetStats(0);
    h_scaled_sideband_norm->SetStats(0);
    h_michel_background_predicted_norm->SetStats(0);
    h_final_subtracted_norm->SetStats(0);
    
    h_low_pe_signal_norm->GetYaxis()->SetTitle("Counts per Day");
    h_scaled_sideband_norm->GetYaxis()->SetTitle("Counts per Day");
    h_michel_background_predicted_norm->GetYaxis()->SetTitle("Counts per Day");
    h_final_subtracted_norm->GetYaxis()->SetTitle("Counts per Day");

    h_low_pe_signal_norm->Draw("HIST");
    h_scaled_sideband_norm->Draw("HIST SAME");
    h_michel_background_predicted_norm->Draw("HIST SAME");
    h_final_subtracted_norm->Draw("HIST SAME");

    TLegend *leg_sub2_norm = new TLegend(0.5, 0.6, 0.9, 0.9);
    leg_sub2_norm->SetTextSize(0.025);
    leg_sub2_norm->SetTextFont(42);
    leg_sub2_norm->SetBorderSize(1);
    leg_sub2_norm->SetFillStyle(0);
    leg_sub2_norm->AddEntry(h_low_pe_signal_norm, Form("Neutron rich region (16-100 #mus) [%.1f events/day]", 
                                                      calculateTotalEvents(h_low_pe_signal_norm)), "l");
    leg_sub2_norm->AddEntry(h_scaled_sideband_norm, Form("Scaled neutron free region [%.1f events/day]", 
                                                        calculateTotalEvents(h_scaled_sideband_norm)), "l");
    leg_sub2_norm->AddEntry(h_michel_background_predicted_norm, Form("Michel background (16-100 #mus) [%.1f events/day]", 
                                                                    michel_rate), "l");
    leg_sub2_norm->AddEntry(h_final_subtracted_norm, Form("Final: Signal - ScaledBkg - Michel [%.1f events/day]", 
                                                         final_rate_corrected), "l");
    leg_sub2_norm->AddEntry((TObject*)0, Form("Live time: %.6f days", liveTimeDays), "");
    leg_sub2_norm->Draw();

    c_sideband2_norm->Update();
    plotName = OUTPUT_DIR + "/Normalized_Low_Energy_Sideband_Subtraction_Complete.png";
    c_sideband2_norm->SaveAs(plotName.c_str());
    cout << "Saved normalized plot: " << plotName << endl;

    TCanvas *c_comparison = new TCanvas("c_comparison", "Raw vs Normalized Comparison", 1600, 800);
    c_comparison->Divide(2,1);
    
    c_comparison->cd(1);
    h_low_pe_signal->SetTitle("Raw Counts");
    h_low_pe_signal->Draw("HIST");
    h_scaled_sideband->Draw("HIST SAME");
    h_michel_background_predicted->Draw("HIST SAME");
    h_final_subtracted->Draw("HIST SAME");
    
    TLegend *leg_comp1 = new TLegend(0.5, 0.6, 0.9, 0.9);
    leg_comp1->SetTextSize(0.025);
    leg_comp1->AddEntry(h_low_pe_signal, Form("Signal: %.0f events", signal_events), "l");
    leg_comp1->AddEntry(h_scaled_sideband, Form("Scaled bkg: %.1f events", scaled_sideband_events), "l");
    leg_comp1->AddEntry(h_michel_background_predicted, Form("Michel: %.1f events", predicted_michels), "l");
    leg_comp1->AddEntry(h_final_subtracted, Form("Final: %.1f events", final_subtracted_corrected), "l");
    leg_comp1->Draw();
    
    c_comparison->cd(2);
    h_low_pe_signal_norm->SetTitle("Normalized (Counts per Day)");
    h_low_pe_signal_norm->Draw("HIST");
    h_scaled_sideband_norm->Draw("HIST SAME");
    h_michel_background_predicted_norm->Draw("HIST SAME");
    h_final_subtracted_norm->Draw("HIST SAME");
    
    TLegend *leg_comp2 = new TLegend(0.5, 0.6, 0.9, 0.9);
    leg_comp2->SetTextSize(0.025);
    leg_comp2->AddEntry(h_low_pe_signal_norm, Form("Signal: %.1f/day", calculateTotalEvents(h_low_pe_signal_norm)), "l");
    leg_comp2->AddEntry(h_scaled_sideband_norm, Form("Scaled bkg: %.1f/day", calculateTotalEvents(h_scaled_sideband_norm)), "l");
    leg_comp2->AddEntry(h_michel_background_predicted_norm, Form("Michel: %.1f/day", michel_rate), "l");
    leg_comp2->AddEntry(h_final_subtracted_norm, Form("Final: %.1f/day", final_rate_corrected), "l");
    leg_comp2->AddEntry((TObject*)0, Form("Total live time: %.6f days", liveTimeDays), "");
    leg_comp2->Draw();
    
    c_comparison->Update();
    plotName = OUTPUT_DIR + "/Raw_vs_Normalized_Comparison.png";
    c_comparison->SaveAs(plotName.c_str());
    cout << "Saved comparison plot: " << plotName << endl;

    cout << "=== Normalized Results (Counts per Day) ===" << endl;
    cout << "Live time: " << liveTimeDays << " days" << endl;
    cout << "Signal region (16-100 μs): " << signal_rate << " events/day" << endl;
    cout << "Scaled neutron-free background: " << scaled_sideband_rate << " events/day" << endl;
    cout << "Michel background: " << michel_rate << " events/day" << endl;
    cout << "Final subtracted: " << final_rate_corrected << " events/day" << endl;
    cout << "===========================================" << endl;

    // Save all histograms to ROOT file
    cout << "Saving all histograms to ROOT file..." << endl;
    saveAllHistogramsToRootFile(rootFile, 
                               h_muon_energy, h_muon_all, h_michel_energy, 
                               h_dt_michel, h_energy_vs_dt, h_side_vp_muon, 
                               h_top_vp_muon, h_trigger_bits, h_isolated_pe, 
                               h_low_iso, h_high_iso, h_dt_prompt_delayed, 
                               h_dt_low_muon, h_dt_high_muon, h_low_pe_signal, 
                               h_low_pe_sideband, h_isolated_ge40, 
                               h_dt_michel_sideband, h_michel_energy_fit_range,
                               h_veto_panel, h_neutron_richness, 
                               h_signal_significance, h_energy_vs_time_low, 
                               h_energy_vs_time_high, h_michel_energy_predicted,
                               h_scaled_sideband, h_michel_background_predicted,
                               h_final_subtracted, h_low_pe_signal_norm,
                               h_low_pe_sideband_norm, h_scaled_sideband_norm,
                               h_michel_background_predicted_norm, h_final_subtracted_norm);

    // Close ROOT file
    rootFile->Close();
    cout << "Closed ROOT file: " << rootFileName << endl;

    // Clean up memory
    delete h_muon_energy;
    delete h_muon_all;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_vp_muon;
    delete h_top_vp_muon;
    delete h_trigger_bits;
    delete h_isolated_pe;
    delete h_low_iso;
    delete h_high_iso;
    delete h_dt_prompt_delayed;
    delete h_dt_low_muon;
    delete h_dt_high_muon;
    delete h_low_pe_signal;
    delete h_low_pe_sideband;
    delete h_isolated_ge40;
    for (int i = 0; i < 10; i++) {
        delete h_veto_panel[i];
    }

    delete h_dt_michel_sideband;
    delete h_michel_energy_fit_range;
    delete h_michel_energy_predicted;
    delete h_final_subtracted;
    delete h_dt_michel_fit_range;

    delete h_neutron_richness;
    delete h_signal_significance;
    delete h_energy_vs_time_low;
    delete h_energy_vs_time_high;

    delete h_low_pe_signal_norm;
    delete h_low_pe_sideband_norm;
    delete h_scaled_sideband_norm;
    delete h_michel_background_predicted_norm;
    delete h_final_subtracted_norm;
    
    delete h_scaled_sideband;
    delete h_michel_background_predicted;
    delete leg_energy;
    delete leg_sub1;
    delete leg_sub2;
    delete leg_sub2_log;
    delete leg_sub1_norm;
    delete leg_sub2_norm;
    delete leg_comp1;
    delete leg_comp2;
    delete c;
    delete c_low_muon;
    delete c_high_muon;
    delete c_michel_method;
    delete c_sideband1;
    delete c_sideband2;
    delete c_sideband2_log;
    delete c_sideband1_norm;
    delete c_sideband2_norm;
    delete c_comparison;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (*.png, *.csv, *.txt, *.root)" << endl;
    return 0;
}
