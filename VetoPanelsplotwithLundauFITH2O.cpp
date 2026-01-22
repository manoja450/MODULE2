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
#include <TPad.h>
#include <TFitResult.h>

using std::cout;
using std::endl;
using namespace std;

// Constants for veto panel analysis
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;
const int BS_UNCERTAINTY = 5;
const int EV61_THRESHOLD = 1200;
const double MUON_ENERGY_THRESHOLD = 50;
const int ADCSIZE = 45;

// Generate unique output directory
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./MuonThresholdDetermination_" + getTimestamp();

// UPDATED: Veto panel thresholds - channels 12-13 are top panels, 14-21 are side panels
const std::vector<double> SIDE_VP_THRESHOLDS = {1200, 1800, 1000, 1000, 1000, 800, 1100, 1100}; // Channels 14-21
const double TOP_VP_THRESHOLD = 1000; // Channels 12-13

// UPDATED: Color scheme for veto panel groups
// Group 1: Top Panels 12-13 (Red shades)
// Group 2: Side Panels 14-17 (Blue shades)
// Group 3: Side Panels 18-21 (Green shades)
const int VETO_PANEL_COLORS[10] = {
    kRed,         // Panel 12 (TOP)
    kRed,         // Panel 13 (TOP)
    kBlue,        // Panel 14 (SIDE)
    kBlue,        // Panel 15 (SIDE)
    kBlue,        // Panel 16 (SIDE)
    kBlue,        // Panel 17 (SIDE)
    kGreen,       // Panel 18 (SIDE)
    kGreen,       // Panel 19 (SIDE)
    kGreen,       // Panel 20 (SIDE)
    kGreen        // Panel 21 (SIDE)
};

// Forward declarations
void createSummaryCanvas(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, TF1* fit_functions[9], const string& outputDir);
void createSummaryCanvasNoFit(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir);
TF1* createTopVetoCombinedPlot(TH1D* h_top_veto_combined, const string& outputDir);
void createTopVetoCombinedPlotNoFit(TH1D* h_top_veto_combined, const string& outputDir);
void createVetoPanelPlots(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir);
void createVetoPanelPlotsNoFit(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir);

// Pulse structure
struct pulse {
    double start;
    double end;
    double peak;
    double energy;
    double number;
    bool single;
    bool beam;
    double trigger;
    double side_vp_energy;
    double top_vp_energy;
    double all_vp_energy;
    double last_muon_time;
    bool is_muon;
    bool is_michel;
    bool veto_hit[10];
};

// Temporary pulse structure
struct pulse_temp {
    double start;
    double end;
    double peak;
    double energy;
};

// Landau fit function
Double_t LandauFit(Double_t *x, Double_t *par) {
    // par[0] = normalization
    // par[1] = MPV (Most Probable Value)
    // par[2] = width
    // par[3] = constant background
    
    Double_t landau = TMath::Landau(x[0], par[1], par[2], kTRUE);
    return par[0] * landau + par[3];
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
    return max_count > 1 ? most_common : getAverage(v);
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

// Function to create summary canvas with all 9 plots (WITH FIT)
void createSummaryCanvas(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, TF1* fit_functions[9], const string& outputDir) {
    cout << "\n=== Creating Summary Canvas with All 9 Veto Panel Plots (With Landau Fits) ===" << endl;
    
    // Create a large canvas divided into 3x3 grid
    TCanvas *c_summary = new TCanvas("c_veto_summary", 
                                    "Summary - All Veto Panel Energy Distributions with Landau Fits", 
                                    2400, 1800);
    
    // Divide canvas into 3x3 grid
    c_summary->Divide(3, 3);
    
    // Configure style for summary plots
    gStyle->SetOptStat(1110); // Only show entries, mean, std dev
    gStyle->SetOptFit(0);     // Hide fit statistics completely
    
    // Debug output
    cout << "Summary canvas layout:" << endl;
    cout << "  Position 1: Combined top veto panels 12+13 (" << h_top_veto_combined->GetEntries() << " entries)" << endl;
    
    // POSITION 1: COMBINED TOP VETO PANELS (12+13)
    c_summary->cd(1);
    gPad->SetLogy();
    
    // Configure combined top veto histogram
    h_top_veto_combined->SetLineColor(kBlack);
    h_top_veto_combined->SetLineWidth(1);
    h_top_veto_combined->SetFillColor(kRed);
    h_top_veto_combined->SetFillStyle(3003);
    
    // Draw histogram
    h_top_veto_combined->Draw("hist");
    
    // Add fit function if available
    if (fit_functions[8]) {
        fit_functions[8]->SetLineColor(kRed);
        fit_functions[8]->SetLineWidth(2);
        fit_functions[8]->Draw("same");
    }
    
    gPad->Update();
    
    // POSITIONS 2-9: SIDE VETO PANELS (14-21)
    int plot_position = 2; // Start at position 2
    
    for (int i = 2; i < 10; i++) {  // i=2 corresponds to panel 14
        if (plot_position > 9) break;
        
        if (h_veto_panel[i]->GetEntries() < 10) {
            cout << "  Skipping panel " << i+12 << " - insufficient entries" << endl;
            plot_position++;
            continue;
        }
        
        cout << "  Position " << plot_position << ": Panel " << i+12 << " (" << h_veto_panel[i]->GetEntries() << " entries)" << endl;
        
        c_summary->cd(plot_position);
        plot_position++;
        
        gPad->SetLogy();
        
        // Configure histogram
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(1);
        h_veto_panel[i]->SetFillColor(VETO_PANEL_COLORS[i]);
        h_veto_panel[i]->SetFillStyle(3003);
        
        // Draw histogram
        h_veto_panel[i]->Draw("hist");
        
        // Add fit function if available
        if (fit_functions[i-2]) {
            fit_functions[i-2]->SetLineColor(kRed);
            fit_functions[i-2]->SetLineWidth(2);
            fit_functions[i-2]->Draw("same");
        }
        
        gPad->Update();
    }
    
    // Update and save canvas
    c_summary->Update();
    
    string summaryName = outputDir + "/All_Veto_Panels_Summary_WithFits.png";
    c_summary->SaveAs(summaryName.c_str());
    cout << "Saved summary canvas with fits: " << summaryName << endl;
    
    string summaryPdf = outputDir + "/All_Veto_Panels_Summary_WithFits.pdf";
    c_summary->SaveAs(summaryPdf.c_str());
    
    delete c_summary;
    cout << "=== Summary Canvas With Fits Complete ===" << endl;
}

// Function to create summary canvas with all 9 plots (WITHOUT FIT)
void createSummaryCanvasNoFit(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Summary Canvas with All 9 Veto Panel Plots (Energy Deposition Only) ===" << endl;
    
    // Create a large canvas divided into 3x3 grid
    TCanvas *c_summary = new TCanvas("c_veto_summary_nofit", 
                                    "Summary - All Veto Panel Energy Distributions", 
                                    2400, 1800);
    
    // Divide canvas into 3x3 grid
    c_summary->Divide(3, 3);
    
    // Configure style for summary plots
    gStyle->SetOptStat(1110); // Only show entries, mean, std dev
    gStyle->SetOptFit(0);     // Hide fit statistics completely
    
    // POSITION 1: COMBINED TOP VETO PANELS (12+13)
    c_summary->cd(1);
    gPad->SetLogy();
    
    // Configure combined top veto histogram
    h_top_veto_combined->SetLineColor(kBlack);
    h_top_veto_combined->SetLineWidth(1);
    h_top_veto_combined->SetFillColor(kRed);
    h_top_veto_combined->SetFillStyle(3003);
    
    // Draw histogram
    h_top_veto_combined->Draw("hist");
    
    gPad->Update();
    
    // POSITIONS 2-9: SIDE VETO PANELS (14-21)
    int plot_position = 2;
    
    for (int i = 2; i < 10; i++) {  // i=2 corresponds to panel 14
        if (plot_position > 9) break;
        
        if (h_veto_panel[i]->GetEntries() < 10) {
            cout << "Skipping veto panel " << i+12 << " in summary (no fit) - insufficient entries" << endl;
            plot_position++;
            continue;
        }
        
        c_summary->cd(plot_position);
        plot_position++;
        
        gPad->SetLogy();
        
        // Configure histogram
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(1);
        h_veto_panel[i]->SetFillColor(VETO_PANEL_COLORS[i]);
        h_veto_panel[i]->SetFillStyle(3003);
        
        // Draw histogram
        h_veto_panel[i]->Draw("hist");
        
        gPad->Update();
    }
    
    // Update and save canvas
    c_summary->Update();
    
    string summaryName = outputDir + "/All_Veto_Panels_Summary_EnergyOnly.png";
    c_summary->SaveAs(summaryName.c_str());
    cout << "Saved summary canvas without fits: " << summaryName << endl;
    
    string summaryPdf = outputDir + "/All_Veto_Panels_Summary_EnergyOnly.pdf";
    c_summary->SaveAs(summaryPdf.c_str());
    
    delete c_summary;
    cout << "=== Summary Canvas Without Fits Complete ===" << endl;
}

// Function to create combined top veto plot with fit
TF1* createTopVetoCombinedPlot(TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Combined Top Veto Plot (With Landau Fit) ===" << endl;
    cout << "Entries in combined histogram: " << h_top_veto_combined->GetEntries() << endl;
    
    if (h_top_veto_combined->GetEntries() < 10) {
        cout << "WARNING: Combined top veto plot has very few entries: " 
             << h_top_veto_combined->GetEntries() << endl;
        return nullptr;
    }
    
    TCanvas *c = new TCanvas("c_top_veto_combined", 
                            "Combined Top Veto Panels 12+13 - Muon Energy Deposition with Landau Fit", 
                            1200, 800);
    
    // Configure style
    gStyle->SetOptStat(1110); // Only show entries, mean, std dev
    gStyle->SetOptFit(0);     // Hide fit statistics completely
    
    // Set log scale
    c->SetLogy();
    
    // Configure histogram
    h_top_veto_combined->SetLineColor(kBlack);
    h_top_veto_combined->SetLineWidth(2);
    h_top_veto_combined->SetFillColor(kRed);
    h_top_veto_combined->SetFillStyle(3003);
    
    // Draw histogram
    h_top_veto_combined->Draw("hist");
    
    // Get histogram properties
    double hist_max = h_top_veto_combined->GetMaximum();
    double x_min = h_top_veto_combined->GetXaxis()->GetXmin();
    double x_max = h_top_veto_combined->GetXaxis()->GetXmax();
    
    // Create temporary histogram for fitting (above threshold only)
    TH1D* h_top_veto_fit = new TH1D("h_top_veto_fit", "Temporary fit histogram", 
                                   h_top_veto_combined->GetNbinsX(), 
                                   h_top_veto_combined->GetXaxis()->GetXmin(),
                                   h_top_veto_combined->GetXaxis()->GetXmax());
    
    // Copy only bins ABOVE THRESHOLD
    int threshold_bin = h_top_veto_combined->FindBin(TOP_VP_THRESHOLD);
    int last_bin = h_top_veto_combined->GetNbinsX();
    
    for (int bin = threshold_bin; bin <= last_bin; bin++) {
        h_top_veto_fit->SetBinContent(bin, h_top_veto_combined->GetBinContent(bin));
        h_top_veto_fit->SetBinError(bin, h_top_veto_combined->GetBinError(bin));
    }
    
    cout << "Original histogram entries: " << h_top_veto_combined->GetEntries() << endl;
    cout << "Fit histogram entries (above threshold): " << h_top_veto_fit->GetEntries() << endl;
    cout << "Threshold: " << TOP_VP_THRESHOLD << " ADC" << endl;
    cout << "Fit range: " << TOP_VP_THRESHOLD << " to " << x_max << " ADC" << endl;
    
    // Get properties from fit histogram
    double fit_hist_max = h_top_veto_fit->GetMaximum();
    double fit_mean = h_top_veto_fit->GetMean();
    double fit_rms = h_top_veto_fit->GetRMS();
    int fit_max_bin = h_top_veto_fit->GetMaximumBin();
    double mpv_guess = h_top_veto_fit->GetBinCenter(fit_max_bin);
    
    // Estimate background
    double background_guess = 0;
    int n_bkg_bins = 5;
    int start_bkg_bin = h_top_veto_fit->FindBin(TOP_VP_THRESHOLD);
    for (int i = start_bkg_bin; i < start_bkg_bin + n_bkg_bins; i++) {
        background_guess += h_top_veto_fit->GetBinContent(i);
    }
    background_guess /= n_bkg_bins;
    
    cout << "Fit histogram info (above threshold only):" << endl;
    cout << "  Max bin content: " << fit_hist_max << endl;
    cout << "  Mean: " << fit_mean << " ADC" << endl;
    cout << "  RMS: " << fit_rms << " ADC" << endl;
    cout << "  MPV guess: " << mpv_guess << " ADC" << endl;
    cout << "  Background guess: " << background_guess << endl;
    
    // Create Landau fit function
    TF1 *landauFit = new TF1("landauFit_top_combined", LandauFit, x_min, x_max, 4);
    
    // Set initial parameters
    double norm_guess = (fit_hist_max - background_guess) * fit_rms * 2.5;
    double width_guess = fit_rms * 0.8;
    
    landauFit->SetParameters(norm_guess, mpv_guess, width_guess, background_guess);
    landauFit->SetParNames("Norm", "MPV", "Width", "Background");
    landauFit->SetParLimits(0, norm_guess * 0.1, norm_guess * 10);
    landauFit->SetParLimits(1, fit_mean - fit_rms, fit_mean + fit_rms * 2);
    landauFit->SetParLimits(2, width_guess * 0.1, width_guess * 5);
    landauFit->SetParLimits(3, 0, fit_hist_max * 0.5);
    
    landauFit->SetLineColor(kRed);
    landauFit->SetLineWidth(3);
    landauFit->SetNpx(1000);
    
    cout << "Initial fit parameters:" << endl;
    cout << "  Norm: " << norm_guess << endl;
    cout << "  MPV: " << mpv_guess << endl;
    cout << "  Width: " << width_guess << endl;
    cout << "  Background: " << background_guess << endl;
    
    // Perform fit
    cout << "Fitting combined top veto panels (above threshold only)..." << endl;
    Int_t fitStatus = h_top_veto_fit->Fit(landauFit, "SRLN", "", TOP_VP_THRESHOLD, x_max);
    
    // Get fit results
    double mpv = landauFit->GetParameter(1);
    double width = landauFit->GetParameter(2);
    
    // Create drawing function
    TF1 *landauDraw = new TF1("landauDraw_top_combined", LandauFit, x_min, x_max, 4);
    for (int i = 0; i < 4; i++) {
        landauDraw->SetParameter(i, landauFit->GetParameter(i));
    }
    
    landauDraw->SetLineColor(kRed);
    landauDraw->SetLineWidth(3);
    landauDraw->SetNpx(1000);
    
    // Draw fit curve
    landauDraw->Draw("same");
    
    // Update canvas
    c->Update();
    
    // Save plot
    string plotName = outputDir + "/Top_Veto_Panels_12_13_Combined_LandauFit.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved combined top veto plot with fit: " << plotName << endl;
    
    string pdfName = outputDir + "/Top_Veto_Panels_12_13_Combined_LandauFit.pdf";
    c->SaveAs(pdfName.c_str());
    
    // Print fit results
    if (fitStatus == 0) {
        double mpv_err = landauFit->GetParError(1);
        double chi2 = landauFit->GetChisquare();
        double ndf = landauFit->GetNDF();
        double chi2_ndf = (ndf > 0) ? chi2 / ndf : 0;
        
        cout << "=== Combined Top Veto Panels Fit Results ===" << endl;
        cout << "  MPV = " << mpv << " ± " << mpv_err << " ADC" << endl;
        cout << "  Width = " << width << " ADC" << endl;
        cout << "  Norm = " << landauFit->GetParameter(0) << endl;
        cout << "  Background = " << landauFit->GetParameter(3) << endl;
        cout << "  χ²/NDF = " << chi2_ndf << endl;
        cout << "  Fit Entries = " << h_top_veto_fit->GetEntries() << " (above threshold)" << endl;
        cout << "  Total Entries = " << h_top_veto_combined->GetEntries() << " (all data)" << endl;
        cout << "  Fit Range = " << TOP_VP_THRESHOLD << " - " << x_max << " ADC" << endl;
        cout << "  Threshold = " << TOP_VP_THRESHOLD << " ADC" << endl;
        cout << "=====================================" << endl;
    } else {
        cout << "Fit failed with status: " << fitStatus << endl;
    }
    
    delete landauFit;
    delete h_top_veto_fit;
    delete c;
    
    cout << "=== Combined Top Veto Plot With Fit Complete ===" << endl;
    
    return landauDraw;
}

// Function to create combined top veto plot without fit
void createTopVetoCombinedPlotNoFit(TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Combined Top Veto Plot (Energy Deposition Only) ===" << endl;
    cout << "Entries in combined histogram: " << h_top_veto_combined->GetEntries() << endl;
    
    if (h_top_veto_combined->GetEntries() < 10) {
        cout << "WARNING: Combined top veto plot has very few entries: " 
             << h_top_veto_combined->GetEntries() << endl;
        return;
    }
    
    TCanvas *c = new TCanvas("c_top_veto_combined_nofit", 
                            "Top Veto Panels 12+13 - Energy Deposition", 
                            1200, 800);
    
    // Configure style
    gStyle->SetOptStat(1110); // Only show entries, mean, std dev
    gStyle->SetOptFit(0);     // Hide fit statistics completely
    
    // Set log scale
    c->SetLogy();
    
    // Configure histogram
    h_top_veto_combined->SetLineColor(kBlack);
    h_top_veto_combined->SetLineWidth(2);
    h_top_veto_combined->SetFillColor(kRed);
    h_top_veto_combined->SetFillStyle(3003);
    
    // Draw histogram
    h_top_veto_combined->Draw("hist");
    
    // Update canvas
    c->Update();
    
    // Save plot
    string plotName = outputDir + "/Top_Veto_Panels_12_13_Combined_EnergyOnly.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved combined top veto plot without fit: " << plotName << endl;
    
    string pdfName = outputDir + "/Top_Veto_Panels_12_13_Combined_EnergyOnly.pdf";
    c->SaveAs(pdfName.c_str());
    
    delete c;
    cout << "=== Combined Top Veto Plot Without Fit Complete ===" << endl;
}

// Create individual veto panel plots with fits
void createVetoPanelPlots(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Veto Panel Plots (With Landau Fits) ===" << endl;
    
    // Check contents of all veto panels
    cout << "Veto Panel Entries:" << endl;
    for (int i = 0; i < 10; i++) {
        const char* panel_type = (i < 2) ? "TOP" : "SIDE";
        cout << "  Panel " << i+12 << " (" << panel_type << "): " << h_veto_panel[i]->GetEntries() << " entries" << endl;
    }
    cout << "Combined Top (12+13): " << h_top_veto_combined->GetEntries() << " entries" << endl;
    
    // Array to store fit functions
    TF1* fit_functions[9] = {nullptr}; // 8 side panels + 1 combined top
    
    // Create individual plots for SIDE veto panels (14-21)
    for (int i = 2; i < 10; i++) {  // Start from index 2 for panel 14
        if (h_veto_panel[i]->GetEntries() < 10) {
            cout << "Skipping veto panel " << i+12 << " - insufficient entries: " 
                 << h_veto_panel[i]->GetEntries() << endl;
            continue;
        }
        
        cout << "Creating plot for veto panel " << i+12 << " (with fit)..." << endl;
        
        TCanvas *c = new TCanvas(Form("c_veto_%d", i+12), 
                                Form("Veto Panel %d - Muon Energy Deposition with Landau Fit", i+12), 
                                1200, 900);
        
        // Configure style
        gStyle->SetOptStat(1110); // Only show entries, mean, std dev
        gStyle->SetOptFit(0);     // Hide fit statistics completely
        
        // Set log scale
        c->SetLogy();
        
        // Configure histogram
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->SetFillColor(VETO_PANEL_COLORS[i]);
        h_veto_panel[i]->SetFillStyle(3003);
        
        // Draw histogram
        h_veto_panel[i]->Draw("hist");
        
        // Get histogram properties
        double hist_max = h_veto_panel[i]->GetMaximum();
        double x_min = h_veto_panel[i]->GetXaxis()->GetXmin();
        double x_max = h_veto_panel[i]->GetXaxis()->GetXmax();
        
        // Create temporary histogram for fitting
        double threshold = SIDE_VP_THRESHOLDS[i-2]; // i-2 because thresholds start at panel 14
        TH1D* h_veto_fit = new TH1D(Form("h_veto_fit_%d", i+12), "Temporary fit histogram", 
                                   h_veto_panel[i]->GetNbinsX(), 
                                   h_veto_panel[i]->GetXaxis()->GetXmin(),
                                   h_veto_panel[i]->GetXaxis()->GetXmax());
        
        // Copy only bins ABOVE THRESHOLD
        int threshold_bin = h_veto_panel[i]->FindBin(threshold);
        int last_bin = h_veto_panel[i]->GetNbinsX();
        
        for (int bin = threshold_bin; bin <= last_bin; bin++) {
            h_veto_fit->SetBinContent(bin, h_veto_panel[i]->GetBinContent(bin));
            h_veto_fit->SetBinError(bin, h_veto_panel[i]->GetBinError(bin));
        }
        
        // Get properties from fit histogram
        double fit_hist_max = h_veto_fit->GetMaximum();
        double fit_mean = h_veto_fit->GetMean();
        double fit_rms = h_veto_fit->GetRMS();
        int fit_max_bin = h_veto_fit->GetMaximumBin();
        double mpv_guess = h_veto_fit->GetBinCenter(fit_max_bin);
        
        // Estimate background
        double background_guess = 0;
        int n_bkg_bins = 5;
        int start_bkg_bin = h_veto_fit->FindBin(threshold);
        for (int j = start_bkg_bin; j < start_bkg_bin + n_bkg_bins; j++) {
            background_guess += h_veto_fit->GetBinContent(j);
        }
        background_guess /= n_bkg_bins;
        
        // Create Landau fit function
        TF1 *landauFit = new TF1(Form("landauFit_%d", i+12), LandauFit, x_min, x_max, 4);
        
        // Set initial parameters
        double norm_guess = (fit_hist_max - background_guess) * fit_rms * 2.5;
        double width_guess = fit_rms * 0.8;
        
        landauFit->SetParameters(norm_guess, mpv_guess, width_guess, background_guess);
        landauFit->SetParNames("Norm", "MPV", "Width", "Background");
        landauFit->SetParLimits(0, norm_guess * 0.1, norm_guess * 10);
        landauFit->SetParLimits(1, fit_mean - fit_rms, fit_mean + fit_rms * 2);
        landauFit->SetParLimits(2, width_guess * 0.1, width_guess * 5);
        landauFit->SetParLimits(3, 0, fit_hist_max * 0.5);
        
        landauFit->SetLineColor(kRed);
        landauFit->SetLineWidth(3);
        landauFit->SetNpx(1000);
        
        // Perform fit
        Int_t fitStatus = h_veto_fit->Fit(landauFit, "SRLN", "", threshold, x_max);
        
        // Create drawing function
        TF1 *landauDraw = new TF1(Form("landauDraw_%d", i+12), LandauFit, x_min, x_max, 4);
        for (int j = 0; j < 4; j++) {
            landauDraw->SetParameter(j, landauFit->GetParameter(j));
        }
        landauDraw->SetLineColor(kRed);
        landauDraw->SetLineWidth(3);
        landauDraw->SetNpx(1000);
        
        // Store fit function
        fit_functions[i-2] = new TF1(*landauDraw);
        
        // Draw fit curve
        landauDraw->Draw("same");
        
        // Update canvas
        c->Update();
        
        // Save plot
        string plotName = outputDir + Form("/Veto_Panel_%d_LandauFit.png", i+12);
        c->SaveAs(plotName.c_str());
        cout << "Saved veto panel plot with fit: " << plotName << endl;
        
        // Print fit results
        if (fitStatus == 0) {
            double mpv = landauFit->GetParameter(1);
            double mpv_err = landauFit->GetParError(1);
            cout << "Veto Panel " << i+12 << " - MPV: " << mpv << " ± " << mpv_err 
                 << ", Fit Range: " << threshold << " - " << x_max << " ADC" << endl;
        }
        
        delete landauFit;
        delete landauDraw;
        delete h_veto_fit;
        delete c;
    }

    // Create combined top veto plot
    cout << "\nCreating combined top veto plot (with fit)..." << endl;
    fit_functions[8] = createTopVetoCombinedPlot(h_top_veto_combined, outputDir);
    
    // Create summary canvas
    cout << "\nCreating summary canvas with all 9 veto panel plots (with fits)..." << endl;
    createSummaryCanvas(h_veto_panel, h_top_veto_combined, fit_functions, outputDir);
    
    // Clean up
    for (int i = 0; i < 9; i++) {
        if (fit_functions[i]) delete fit_functions[i];
    }
}

// Create individual veto panel plots without fits
void createVetoPanelPlotsNoFit(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Veto Panel Plots (Energy Deposition Only) ===" << endl;
    
    // Create individual plots for SIDE veto panels (14-21)
    for (int i = 2; i < 10; i++) {  // Start from index 2 for panel 14
        if (h_veto_panel[i]->GetEntries() < 10) {
            cout << "Skipping veto panel " << i+12 << " (no fit) - insufficient entries: " 
                 << h_veto_panel[i]->GetEntries() << endl;
            continue;
        }
        
        cout << "Creating plot for veto panel " << i+12 << " (energy deposition only)..." << endl;
        
        TCanvas *c = new TCanvas(Form("c_veto_%d_nofit", i+12), 
                                Form("Veto Panel %d - Energy Deposition", i+12), 
                                1200, 800);
        
        // Configure style
        gStyle->SetOptStat(1110); // Only show entries, mean, std dev
        gStyle->SetOptFit(0);     // Hide fit statistics completely
        
        // Set log scale
        c->SetLogy();
        
        // Configure histogram
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->SetFillColor(VETO_PANEL_COLORS[i]);
        h_veto_panel[i]->SetFillStyle(3003);
        
        // Draw histogram
        h_veto_panel[i]->Draw("hist");
        
        // Update canvas
        c->Update();
        
        // Save plot
        string plotName = outputDir + Form("/Veto_Panel_%d_EnergyOnly.png", i+12);
        c->SaveAs(plotName.c_str());
        cout << "Saved veto panel plot without fit: " << plotName << endl;
        
        delete c;
    }

    // Create combined top veto plot without fit
    cout << "\nCreating combined top veto plot (energy deposition only)..." << endl;
    createTopVetoCombinedPlotNoFit(h_top_veto_combined, outputDir);
    
    // Create summary canvas without fits
    cout << "\nCreating summary canvas with all 9 veto panel plots (energy deposition only)..." << endl;
    createSummaryCanvasNoFit(h_veto_panel, h_top_veto_combined, outputDir);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    vector<string> inputFiles;
    for (int i = 1; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Create output directory
    createOutputDirectory(OUTPUT_DIR);

    cout << "Veto Panel Muon Analysis" << endl;
    cout << "Output directory: " << OUTPUT_DIR << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    // Statistics counters
    int num_muons = 0;
    int num_events = 0;
    int num_top_veto_hits = 0;

    // Histograms for veto panels (12-21)
    TH1D* h_veto_panel[10];
    const char* veto_names[10] = {
        "Veto Panel 12  - Energy Deposition",       // Channel 12 - TOP
        "Veto Panel 13  - Energy Deposition",       // Channel 13 - TOP
        "Veto Panel 14  - Energy Deposition",      // Channel 14 - SIDE
        "Veto Panel 15  - Energy Deposition",      // Channel 15 - SIDE
        "Veto Panel 16  - Energy Deposition",      // Channel 16 - SIDE
        "Veto Panel 17  - Energy Deposition",      // Channel 17 - SIDE
        "Veto Panel 18  - Energy Deposition",      // Channel 18 - SIDE
        "Veto Panel 19  - Energy Deposition",      // Channel 19 - SIDE
        "Veto Panel 20  - Energy Deposition",      // Channel 20 - SIDE
        "Veto Panel 21  - Energy Deposition"       // Channel 21 - SIDE
    };
    
    // Combined histogram for TOP veto panels (12+13)
    TH1D* h_top_veto_combined = new TH1D("h_top_veto_combined", 
                                        "Top Veto Panels 12+13 - Energy Deposition;Integrated Pulse Area;Counts", 
                                        200, 200, 3000);
    
    // Initialize veto panel histograms
    for (int i = 0; i < 10; i++) {
        h_veto_panel[i] = new TH1D(Form("h_veto_panel_%d", i+12), 
                                  Form("%s;Integrated Pulse Area;Counts", veto_names[i]), 
                                  200, 200, 5000);
    }

    // Process each input file
    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
            continue;
        }

        // Declaration of leaf types
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
            num_events++;

            std::vector<double> veto_energies(10, 0);
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            // Process ALL events and store veto panel energies
            for (int iChan = 0; iChan < 23; iChan++) {
                // Fill waveform histogram
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                // Calculate total energy in waveform
                double allPulseEnergy = 0;
                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    allPulseEnergy += h_wf.GetBinContent(iBin);
                }

                // Store energy for veto panels (channels 12-21)
                if (iChan >= 12 && iChan <= 21) {
                    // Apply calibration factors if needed
                    double factor = 1.0;
                    if (iChan == 20) factor = 1.07809; // Example: channel 20 specific factor
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                }

                h_wf.Reset();
            }

            // Calculate PMT energy for muon identification
            double pmt_energy = 0;
            for (int iChan = 0; iChan <= 11; iChan++) {
                double allPulseEnergy = 0;
                for (int i = 0; i < ADCSIZE; i++) {
                    allPulseEnergy += (adcVal[iChan][i] - baselineMean[iChan]);
                }
                pmt_energy += allPulseEnergy;
            }

            // Fill ALL veto panel histograms with ALL data
            for (int i = 0; i < 10; i++) {
                h_veto_panel[i]->Fill(veto_energies[i]);
            }
            
            // Fill combined top veto histogram with MAX of panels 12 and 13
            double max_top_energy = std::max(veto_energies[0], veto_energies[1]); // Indices 0,1 = panels 12,13
            h_top_veto_combined->Fill(max_top_energy);

            // Muon detection using veto panels
            bool veto_hit = false;
            
            // Check SIDE panels (14-21) - indices 2-9 in veto_energies
            for (size_t i = 2; i < 10; i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i-2]) {
                    veto_hit = true;
                    break;
                }
            }
            
            // Check TOP panels (12-13) - indices 0-1 in veto_energies
            if (!veto_hit && (veto_energies[0] > TOP_VP_THRESHOLD || veto_energies[1] > TOP_VP_THRESHOLD)) {
                veto_hit = true;
            }

            if (pmt_energy > MUON_ENERGY_THRESHOLD && veto_hit) {
                num_muons++;
                if (veto_energies[0] > TOP_VP_THRESHOLD || veto_energies[1] > TOP_VP_THRESHOLD) {
                    num_top_veto_hits++;
                }
            }
        }

        cout << "File " << inputFileName << " - Events: " << num_events << ", Muons: " << num_muons 
             << ", Top Veto Hits: " << num_top_veto_hits << endl;
        f->Close();
    }

    // Print statistics
    cout << "\n=== Final Statistics Before Plotting ===" << endl;
    cout << "Total events processed: " << num_events << endl;
    cout << "Total muons identified: " << num_muons << endl;
    cout << "Total top veto hits (panels 12-13): " << num_top_veto_hits << endl;
    cout << "Combined top veto histogram entries: " << h_top_veto_combined->GetEntries() << endl;
    
    for (int i = 0; i < 10; i++) {
        const char* panel_type = (i < 2) ? "TOP" : "SIDE";
        cout << "Veto Panel " << i+12 << " (" << panel_type << "): " 
             << h_veto_panel[i]->GetEntries() << " entries" << endl;
    }
    cout << "=====================================" << endl;

    // Create plots WITH fits
    createVetoPanelPlots(h_veto_panel, h_top_veto_combined, OUTPUT_DIR);
    
    // Create plots WITHOUT fits
    createVetoPanelPlotsNoFit(h_veto_panel, h_top_veto_combined, OUTPUT_DIR);

    // Final summary
    cout << "\n=== Analysis Complete ===" << endl;
    cout << "Total muon events: " << num_muons << endl;
    cout << "Top veto hits (panels 12-13): " << num_top_veto_hits << endl;
    cout << "Combined top veto entries (panels 12+13): " << h_top_veto_combined->GetEntries() << endl;
    cout << "Results saved in: " << OUTPUT_DIR << endl;

    // Clean up
    for (int i = 0; i < 10; i++) delete h_veto_panel[i];
    delete h_top_veto_combined;
    
    return 0;
}
