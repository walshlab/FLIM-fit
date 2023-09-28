%% Normalize 
clear
fileID = fopen('IRF.txt','r');
formatSpec = '%f';
irf_whole = fscanf(fileID,formatSpec);
irf = irf_whole;
irfNorm = irf/max(irf);


%% Deconvolution the image


[ha1, ha2, ht1, ht2] = deconvolutionFit(hFLIM,irfNorm);
[la1, la2, lt1, lt2] =  deconvolutionFit(lFLIM,irfNorm);

a1Norm = a1./(a1+a2);
a1Norm(isnan(a1Norm))=0;
a2Norm = a2./(a1+a2);
a2Norm(isnan(a2Norm))=0;
%% Test with one pixel
x = 20;
y = 15;
ra1 = a1(x,y);
ra2 = a2(x,y);
rt1 = t1(x,y);
rt2 = t2(x,y);
pixelDecayH = hFLIM(x,y,:);
pixelDecayH = pixelDecayH(:);


pixelDecayL = lFLIM(x,y,:)*5;
pixelDecayL = pixelDecayL(:);


plot(pixelDecayH)
hold on
plot(pixelDecayL);

legend('Ideal Decay','Few-Photon Decay')
%%
timeFitting = (1:256)*12.5/256 ;
deconvDecay = deconvlucy(pixelDecayH,irfNorm(70:92),10);
[~, peakPosition] = max(deconvDecay);
fittingPart = [deconvDecay(peakPosition:256)' zeros(1,peakPosition-1)];
decayFit = fit(timeFitting',fittingPart','exp2','StartPoint',[3,-7, 1, -0.6]);
fittingValues = coeffvalues(decayFit);
[fa1, ft1, fa2, ft2] = Calculate_FLIM(fittingValues);

% Compare the fitting result and ground truth
time = (1:256)*12.5/256;
p =(ra1)*exp(-time/rt1)+ (ra2)*exp(-time/rt2);


q = (fa1)*exp(-time/ft1)+ (fa2)*exp(-time/ft2);

plot(p)
hold on
plot(q);

%% Show different fitting results of different photon numbers
subplot(4,3,1)
imagesc(a1Norm)
caxis([0 1]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,2)
imagesc(ha1)
caxis([0 1]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,3)
imagesc(la1)
caxis([0 1]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,4)
imagesc(a2Norm)
caxis([0 0.5]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,5)
imagesc(ha2)
caxis([0 0.5]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,6)
imagesc(la2)
caxis([0 0.5]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,7)
imagesc(t1)
caxis([0 0.6]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,8)
imagesc(ht1)
caxis([0 0.6]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,9)
imagesc(lt1);
caxis([0 0.6]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,10)
imagesc(t2)
caxis([0 4]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,11)
imagesc(ht2)
caxis([0 4]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])

subplot(4,3,12)
imagesc(lt2)
caxis([0 4]);
set(gca,'xtick',[])
set(gca,'ytick',[])
pbaspect([1 1 1])
%%
hDifa1 = imabsdiff(a1Norm,ha1);
hDifa2 = imabsdiff(a2Norm,ha2);
hDift1 = imabsdiff(t1,ht1);
hDift2 = imabsdiff(t2,ht2);

lDifa1 = imabsdiff(a1Norm,la1);
lDifa2 = imabsdiff(a2Norm,la2);
lDift1 = imabsdiff(t1,lt1);
lDift2 = imabsdiff(t2,lt2);
%% Quantify the differences
subplot(2,4,1)
imagesc(hDifa1)
caxis([0 0.3]);
pbaspect([1 1 1])

subplot(2,4,5)
imagesc(lDifa1)
caxis([0 0.3]);
pbaspect([1 1 1])

subplot(2,4,2)
imagesc(hDifa2);
caxis([0 0.3]);
pbaspect([1 1 1])

subplot(2,4,6)
imagesc(lDifa2);
caxis([0 0.3]);
pbaspect([1 1 1])

subplot(2,4,3)
imagesc(hDift1);
caxis([0 0.3]);
pbaspect([1 1 1])

subplot(2,4,7)
imagesc(lDift1);
caxis([0 0.3]);
pbaspect([1 1 1])

subplot(2,4,4)
imagesc(hDift2);
caxis([0 1]);
pbaspect([1 1 1])

subplot(2,4,8)
imagesc(lDift2);
caxis([0 1]);
pbaspect([1 1 1])







function [a1Image, a2Image, t1Image, t2Image] = deconvolutionFit(tpsfImage,irfNorm)
    timeFitting = (1:256)*12.5/256 ;
    % image size  = 28
    a1Image = zeros(28,28);
    a2Image = zeros(28,28);
    t1Image = zeros(28,28);
    t2Image = zeros(28,28);
        for i = 1:28
            for j = 1:28
                pixelDecay = tpsfImage(i,j,:);
                pixelDecay = pixelDecay(:);
                if max(pixelDecay)> 0
                    deconvDecay = deconvlucy(pixelDecay,irfNorm(70:92),10);
                    [~, peakPosition] = max(deconvDecay);
                    fittingPart = [deconvDecay(peakPosition:256)' zeros(1,peakPosition-1)];
                    decayFit = fit(timeFitting',fittingPart','exp2','StartPoint',[3,-7, 1, -0.6]);
                    fittingValues = coeffvalues(decayFit);
                    [fa1, ft1, fa2, ft2] = Calculate_FLIM(fittingValues);
                    a1Image(i,j) = fa1;
                    a2Image(i,j) = fa2;
                    t1Image(i,j) = ft1;
                    t2Image(i,j) = ft2;
                else
                    a1Image(i,j) = 0;
                    a2Image(i,j) = 0;
                    t1Image(i,j) = 0;
                    t2Image(i,j) = 0;
                end   
            end
        end
end

function [a1,t1,a2,t2] = Calculate_FLIM(fittingValues)
% a1 = fittingValues(1)/(fittingValues(1) + fittingValues(3) );
a1 = fittingValues(1);
t1 = -1/fittingValues(2);
% a2 = fittingValues(3)/(fittingValues(1) + fittingValues(3) );
a2 = fittingValues(3);
t2 = -1/fittingValues(4);
end