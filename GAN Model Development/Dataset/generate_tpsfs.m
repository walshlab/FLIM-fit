function [pixelFullDataHigh,pixelFullDataLow] = generate_tpsfs(tau1, tau2, alpha1, alpha2,irf)
% dimension: image * time gate (28x28xnTG)
% irf: unit Instrumental Response Function (sum=1)

m = size(tau1, 1);
n = size(tau1, 2);
% Number of time-points/gates
nTG = 256;

%% 12.5 ns with 256 time points
width = 12.5/256; % Different time-point durations for different apparatus settings
time = (1:1:nTG)*width;
% Pre-allocate memory for each TPSF voxel
pixelFullDataHigh = zeros(m,n,nTG);
pixelFullDataLow = zeros(m,n,nTG);
% Loop over all pixels spatially
irfNorm = irf/max(irf);
for i=1:m
    for j=1:n
%         Only loop at locations from which TPSFs can be created.
        if tau1(i,j)~=0
%             Create initial bi-exponential given the tau1, tau2 and ratio
%             values at the image position (i,j)
            decayHigh = alpha1(i,j)*exp(-time./tau1(i,j))+alpha2(i,j)*exp(-time./tau2(i,j));
            decayLow = alpha1(i,j)/5*exp(-time./tau1(i,j))+alpha2(i,j)/5*exp(-time./tau2(i,j));
%           Grab IRF from library
              

%             Convolve IRF with our exp. decay
            decayHigh = conv(decayHigh,irfNorm);
            decayLow = conv(decayLow,irfNorm);
%             Sample back to the original number of time-points by including random
%             effects due to laser-jitter (point of TPSF ascent).
            rng(10)
            r = rand();
            % Shift -5 to +5 of the decay curve
            if r > .75
                rC =  randi([1,5]);
                decayHigh = decayHigh(1+rC:nTG+rC);
                decayLow = decayLow(1+rC:nTG+rC);
            elseif r < .25
                rC = randi([1,5]);
                decayHigh = [zeros(rC,1); decayHigh(1:nTG-rC)];
                decayLow = [zeros(rC,1); decayLow(1:nTG-rC)];
            else
                decayHigh = decayHigh(1:nTG);
                decayLow = decayLow(1:nTG);
            end
            
%             Multiple the decay by its corresponding intensity value
%             (maximum photon count)
%             Add poisson noise
            finalDecayHigh = round(poissrnd(decayHigh));
            finalDecayLow = round(poissrnd(decayLow));
            
%             Assign the decay to its corresponding pixel location
            pixelFullDataHigh(i,j,:) = finalDecayHigh;
            pixelFullDataLow(i,j,:) = finalDecayLow;
        end
    end
end

end