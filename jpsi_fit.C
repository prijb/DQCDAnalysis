//Getting FWHM from the Voigtian sigma and width (gamma?)
double voigt_FWHM(double width, double sigma){
    double g, l;
    g = 2*sigma*TMath::Sqrt(2*TMath::Log(2));
    l = 2*width;
    return (0.5346*l + TMath::Sqrt(TMath::Power(0.2166*l, 2) + TMath::Power(g, 2)));
}

void jpsi_fit(){

    using namespace RooFit;

    TFile *file = new TFile("histograms.root", "READ");
    TH1D *hist_bpark23 = (TH1D*)file->Get("hist_jpsi_bparking23");
    TH1D *hist_bpark23_orig = (TH1D*)file->Get("hist_jpsi_bparking23_orig");
    TH1D *hist_bpark18 = (TH1D*)file->Get("hist_jpsi_bparking18");
    TH1D *hist_scouting = (TH1D*)file->Get("hist_jpsi_scouting");

    hist_bpark23->SetLineColor(kOrange);
    hist_bpark23_orig->SetLineColor(kGreen);
    hist_bpark18->SetLineColor(kBlue);
    hist_scouting->SetLineColor(kRed);

    //var
    RooRealVar m("m", "SV mass (GeV)", hist_bpark23->GetXaxis()->GetXmin(), hist_bpark23->GetXaxis()->GetXmax());
    
    //signal model
    RooRealVar mean("mean", "mean", 3.1, 2.8, 3.3);
    RooRealVar width("width", "width", 0.2, 0., 1.);
    RooRealVar sigma("sigma", "sigma", 0.2, 0., 1.);

    RooVoigtian signal("signal", "Signal PDF", m, mean, width, sigma);
    //RooGaussian signal("signal", "Signal PDF", m, mean, sigma);

    //background model
    RooRealVar a1("a1","a1",-0.1, -10., 0.);
    RooExponential bkg("bkg", "Background PDF", m, a1);

    //Make total PDF
    RooRealVar nsig("nsig", "#signal events", 1e6, 1e2, 1e10);
    RooRealVar nbkg("nbkg", "#background events", 1e6, 1e2, 1e10);
    RooAddPdf model("model", "sig+bkg", {signal, bkg}, {nsig, nbkg});

    //Get data
    RooDataHist data_bpark23("data_bpark23","data_bpark23",m,Import(*hist_bpark23));
    RooDataHist data_bpark23_orig("data_bpark23_orig","data_bpark23_orig",m,Import(*hist_bpark23_orig));
    RooDataHist data_bpark18("data_bpark18","data_bpark18",m,Import(*hist_bpark18));
    RooDataHist data_scouting("data_scouting","data_scouting",m,Import(*hist_scouting));
    
    //Get result and plot
    auto frame_data = m.frame();
    frame_data->SetTitle("J/Psi peak");

    //Save fit results for printing
    auto fit_bpark23 = model.fitTo(data_bpark23, RooFit::Save(true));
    TCanvas *c1 = new TCanvas();
    auto frame_bpark23 = m.frame();
    frame_bpark23->SetTitle("B-Parking 2023C (muonSV)");
    data_bpark23.plotOn(frame_bpark23);
    data_bpark23.plotOn(frame_data, LineColor(kOrange));
    model.plotOn(frame_bpark23, LineColor(kOrange));
    model.plotOn(frame_data, LineColor(kOrange));
    frame_bpark23->Draw(); 


    auto fit_bpark23_orig = model.fitTo(data_bpark23_orig, RooFit::Save(true));
    TCanvas *c2 = new TCanvas();
    auto frame_bpark23_orig = m.frame();
    frame_bpark23_orig->SetTitle("B-Parking 2023C (SV)");
    data_bpark23_orig.plotOn(frame_bpark23_orig);
    data_bpark23_orig.plotOn(frame_data, LineColor(kGreen));
    model.plotOn(frame_bpark23_orig, LineColor(kGreen));
    model.plotOn(frame_data, LineColor(kGreen));
    frame_bpark23_orig->Draw(); 

    auto fit_bpark18 = model.fitTo(data_bpark18, RooFit::Save(true));
    TCanvas *c3 = new TCanvas();
    auto frame_bpark18 = m.frame();
    frame_bpark18->SetTitle("B-Parking 2018 (muonSV)");
    data_bpark18.plotOn(frame_bpark18);
    data_bpark18.plotOn(frame_data, LineColor(kBlue));
    model.plotOn(frame_bpark18, LineColor(kBlue));
    model.plotOn(frame_data, LineColor(kBlue));
    frame_bpark18->Draw(); 

    auto fit_scouting = model.fitTo(data_scouting, RooFit::Save(true));
    TCanvas *c4 = new TCanvas();
    auto frame_scouting = m.frame();
    frame_scouting->SetTitle("Scouting 2022F (Scouting SV)");
    data_scouting.plotOn(frame_scouting);
    data_scouting.plotOn(frame_data, LineColor(kRed));
    model.plotOn(frame_scouting, LineColor(kRed));
    model.plotOn(frame_data, LineColor(kRed));
    frame_scouting->Draw(); 

    TCanvas *c5 = new TCanvas;
    gStyle->SetOptStat(0);
    //TH1D *h_canvas = (TH1D*)c5->GetPrimitive("Data");
    hist_bpark23->SetAxisRange(5e3, 3e6, "Y");
    hist_bpark23->SetTitle("Secondary vertex mass");
    hist_bpark23->GetXaxis()->SetTitle("Secondary vertex mass (GeV)");
    hist_bpark23->GetYaxis()->SetTitle("Vertices/(0.02 GeV)");
    hist_bpark23->Draw();
    hist_bpark23_orig->Draw("SAME");
    hist_bpark18->Draw("SAME");
    hist_scouting->Draw("SAME");
    frame_data->Draw("SAME");
    
    auto legend = new TLegend(0.7,0.7,0.9,0.9);
    legend->AddEntry(hist_bpark23, "BParking 2023C muonSV");
    legend->AddEntry(hist_bpark23_orig, "BParking 2023C SV");
    legend->AddEntry(hist_bpark18, "BParking 2018 muonSV");
    legend->AddEntry(hist_scouting, "Scouting 2022F SV");
    legend->Draw("SAME");
    c5->SetLogy();
    
    std::vector<std::string> dataset_name{"BParking 2023C muonSV", "BParking 2023C SV", "BParking 2018 muonSV", "Scouting 2022F SV"};
    std::vector<RooFitResult*> fit_results{fit_bpark23, fit_bpark23_orig, fit_bpark18, fit_scouting};

    std::cout << std::endl;
    std::cout << std::endl;

    for(int i=0; i<fit_results.size(); i++){
        RooFitResult *result = (RooFitResult*)fit_results.at(i);
        const RooArgList &params = result->floatParsFinal();
        double s = ((RooRealVar*)params.at(3))->getVal();
        double b = ((RooRealVar*)params.at(2))->getVal();
        std::cout << "Data: "  << dataset_name.at(i) << std::endl;
        std::cout << "J/Psi mass= (" << ((RooRealVar*)params.at(1))->getVal() << " +/- " << 0.5*voigt_FWHM(((RooRealVar*)params.at(5))->getVal(), ((RooRealVar*)params.at(4))->getVal()) << ") GeV" << std::endl;
        std::cout << "Signal yield= " << ((RooRealVar*)params.at(3))->getVal() << " vertices" << std::endl;
        std::cout << "Background yield= " << ((RooRealVar*)params.at(2))->getVal() << " vertices" << std::endl;
        std::cout << "S/sqrt(B)= " << ((s)/TMath::Sqrt(b)) << std::endl;
        std::cout << "Significance= " << TMath::Sqrt(2*(((s+b)*TMath::Log(1 + (s/b))) - s)) << std::endl;
        std::cout << std::endl;
    }

    /*
    std::cout << "BParking 23 (muonSV) fit" << std::endl;
    fit_bpark23->Print();
    std::cout << "BParking 23 (SV) fit" << std::endl;
    fit_bpark23_orig->Print();
    std::cout << "BParking 18 (muonSV) fit" << std::endl;
    fit_bpark18->Print();
    std::cout << "Scouting fit" << std::endl;
    fit_scouting->Print();
    */

}