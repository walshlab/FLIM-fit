%% generate the mnist image masks
load('mnist.mat')
% The MNIST dataset will be load with 4 mat: trainX, trainY, testX, testY

%% Explanation of the resting code
% Two ways to generate the image mask

% 1)  Only select the images low sparsity
    % Here we make use of the trainX testX ((60000 + 10000) X 784),
    % calculate the non-zeorn pixel number in each image, and elimate those
    % <256 images. We need to reshape it to 28 X 28 x 600000, and generate imagemask. Since, The number of these image is only 456, we need to repeat it 22 times to generate the whole dataset

% 2)  Direclty select 10000 images from the trainX dataset
    % Ignoer the sparsity of the image, and make use trainX (60000 X 784),
    % We need to reshape it We need to reshape it to 28 X 28 x 600000, and
    % generate imagemask. Then we randomly select 10000 images from them as
    % our dataset

% 3) Directly use the Minist images ((60000 + 10000) X 784)

%% Section 1: Select the MNIST with low sparsity

% Elimate imags with low sparsity

    sumImage = [trainX;testX];
    noZeroNumberList = sum(sumImage ~= 0,2);
    noZeroNumber = find(noZeroNumberList > 256);
    [totalImageNumber,~] = size(noZeroNumber);

% Generate the MNIST image mask stack
    totalImageStack = zeros(28,28,totalImageNumber);
    for i = 1:totalImageNumber
        image = reshape(sumImage(noZeroNumber(i,1),:),28,28)';
        image(image~=0) = 1;
        totalImageStack(:,:,i) = image;    
    end
% repeat 22 times to get 10000 image stack
    finalImageStack = repmat(totalImageStack,1,1,22);
    finalImageStack(:,:,10001:10032) = [];
    
    
%% Section 2: Directly select the images from trainX of MNIST

% Generate the MNIST image mask stack
    [totalImageNumber,~] = size(trainX);
    totalImageStack = zeros(28,28,totalImageNumber);

    for i = 1: totalImageNumber
        image = reshape(trainX(i,:),28,28)';
        image(image~=0) = 1;
        totalImageStack(:,:,i) = image;    
    end

% Random select 10000 image from the totalImageStack %%
    rng(100)
    randNumber = randperm(60000,10000);
    finalImageStack =  zeros(28,28,10000);
    for i  = 1:10000
        finalImageStack(:,:,i) =  totalImageStack(:,:,randNumber(1,i));
    end
%% Section 3: Directly select the images from trainX and testX of MNIST
    sumImage = [trainX;testX];
    finalImageStack =  zeros(28,28,70000);
    for i  = 1:70000
        image = reshape(sumImage(i,:),28,28)';
        image(image~=0) = 1;
        finalImageStack(:,:,i) =  image;
    end
    
%% 
% load('FLIM_IRF.mat');
fileID = fopen('IRF.txt','r');
formatSpec = '%f';
irf_whole = fscanf(fileID,formatSpec);
irf = irf_whole;
%% Generate the FLIM Data and save
    % nTG = 256;  
    % Number of TPSF voxels to create
        totalTPSPNumber = 70000;
        k = 1;
    
    while k <= totalTPSPNumber
    % Take each mask from the image stack 
        imageMask = finalImageStack(:,:,k);
    % Generate t1, t2 and AR image maps
        [tau1, tau2, alpha1, alpha2] = generate_lifetime( imageMask );
        [measuredFLIMHigh,measuredFLIMLow] = generate_tpsfs(tau1, tau2, alpha1, alpha2,irf);
        
        t1 = tau1;
        t2 = tau2;
        a1 = alpha1;
        a2 = alpha2;
        hFLIM = measuredFLIMHigh;
        lFLIM = measuredFLIMLow;
        
        % Check a representative pixel to see the photon distribution
        a = hFLIM(14,15,:);
        b = lFLIM(14,15,:);
        plot(a(:))
        hold on
        plot(b(:))
    % Making sure sample numbers are assigned like 00001, 00002,.... 01001,
    % 01002, etc.
        if k >=0 && k < 10
            n = ['0000' num2str(k)];
        elseif k >=10 && k<100
            n = ['000' num2str(k)];
        elseif k >=100 && k<1000
            n = ['00' num2str(k)];
        elseif k >=1000 && k<10000
            n = ['0' num2str(k)];
        else
            n = num2str(k);
        end

    % Assign path along with file name.
        savingPath = 'C:\Linghao Hu\Project\FLIM_Fitting\New Dataset\Simulate_Dataset';
        savingName = [savingPath '\' 'FLIMM' '_' n ];

    % Save .mat file. It is important to note the end '-v7.3' - this is one
    % of the more convenient ways to facillitate easy python upload of 
    % matlab-created data.
        save(savingName, 'hFLIM','lFLIM','t1', 't2', 'a1', 'a2','-v7.3');

        k = k+1;
    end    

    






