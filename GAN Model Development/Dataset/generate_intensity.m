%% Generate intensity and FLIM Image    
function intensity = generate_intensity( image )
% generate random intensity for input binary image
    m = size(image, 1);
    n = size(image, 2);
%     random matrix of intensity values possessing values within maximum
%     photon count threshold.
    intensityDistribution = rand(m, n) * 500; % 0 - 500 p.c.
    intensity = intensityDistribution.*image;
end