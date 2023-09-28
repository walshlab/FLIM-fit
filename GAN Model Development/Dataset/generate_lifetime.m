function [tau1, tau2, alpha1, alpha2] = generate_lifetime( image )
% generate random lifetime values for the 28x28 binary image
    
    m = size(image, 1);
    n = size(image, 2);
%     Create randomly generated value matrices for the tau1 and tau2
%     thresholds of interest.
    tau1 = rand(m,n)*0.3 + 0.15; % t1 values between 0.15 - 0.45 ns
    tau2 = rand(m,n)*2 + 1.5; % t2 values between 1.5 - 3.5 ns
    
% Create the decay curve with peak value between 20 - 25    
    peakValue = rand(m,n)*5 + 20;
% Create alpha1 and alpha2
    alpha1Norm = rand(m,n)*0.2 + 0.65; % alpha1 value between 0.65 - 0.85
    alpha2Norm = ones(m,n) - alpha1Norm; % alpha2 value = 1 - alpha1(0.15 - 0.35)

    alpha1 = peakValue.*alpha1Norm;
    alpha2 = peakValue.*alpha2Norm;
    
    tau1 = tau1.*image;
    tau2 = tau2.*image;
    alpha1 = alpha1.*image;
    alpha2 = alpha2.*image;
end