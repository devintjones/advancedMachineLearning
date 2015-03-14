
% Advanced Machine Learning HW1
% Professor Tony Jebara, Columbia University
% Author: Devin Jones
function part4

% mmreader() for matlab 2013b
% use videoReader() for 2014b
vid = mmreader('C:\Users\Devin\Videos\crocs_cut.wmv');

% set playmovie to 1 to watch raw movie
playmovie = 0;
if playmovie == 1
    watchMovie(vid)
end

% convert video to a matrix of black and white images
% will play a video to verify that the data has been transformed properly
bwMat = getBwMat(vid,0);


X = bwMat;

% Kernel Params
kernel = 1; % linear kernel
sigma = 0;  % for rbf kernel
degree = 0; % polynomial kernel degree. not used for part3

% Calculate linear kernal. 
A = calculateAffinityMatrix(X,kernel,sigma,degree);


% Derive distance matrix from kernal
G = getDistanceMat(A);


%populate this matrix with NN & fidelity score for plotting
fidelity = zeros(3,2);

% Iterate over 2-4 nearest neighbors
for NN=5:7
    mveScore = mve2d(G,NN,A);
    fidelity(NN-1,:)= [NN,mveScore]
end

plotTitle = 'MVE Fidelity of Crocs Video';
plotFidelity(fidelity, plotTitle,23)


% Iterate 2D MVE embedding over number of neighbors
function mveScore = mve2d(G,k,A)
    
    % Returns a neighbor graph
    neighbors = calculateNeighborMatrix(G, k,1);

    %parameters for MVE
    targetd = 2;
    tol = 0.99;

    [Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);

    plotTitle = sprintf('MVE embedding: Crocs Product Video. Number of neighbors: %d',k);
    plotEmbedding(Y, neighbors, plotTitle ,k)




function bwMat = getBwMat(vid,saveData)
    vidWidth = vid.Width;
    vidHeight = vid.Height;
    matWidth = vidWidth*vidHeight;

    nFrames = vid.NumberOfFrames;

    bwMat = zeros(matWidth,nFrames);

    for k = 1:nFrames
        frame = rgb2gray(read(vid,k));
        bwMat(:,k) = reshape(frame,1,[]);
    end
    
    disp(sprintf('size of converted movie matrix: %d by %d',size(bwMat)))
    
    for i=1:nFrames; imshow(reshape(bwMat(:,i),vidHeight,vidWidth)/255); pause(1/vid.FrameRate); end;
    if saveData == 1
        save crocsBw.mat bwMat;
    end



% put movie in matrix
function colorMat = getColorMat(vid)

    % matWidth
    vidWidth = vid.Width;
    vidHeight = vid.Height;
    matWidth = vidWidth*vidHeight*3;

    nFrames = vid.NumberOfFrames;

    colorMat= zeros(matWidth,nFrames);

    for k = 1:nFrames
        frame = read(vid,k);
        colorMat(:,k) = reshape(frame,1,[]);
    end
    
    disp(sprintf('size of converted movie matrix: %d by %d',size(colorMat)))
    
    for i=1:nFrames; image(reshape(colorMat(:,i),vidHeight,vidWidth,3)/255); pause(1/vid.FrameRate); end;



% plays video of movie in matlab 2013b
% from the docs
function watchMovie(vid)
    vidWidth = vid.Width;
    vidHeight = vid.Height;
    nFrames = vid.NumberOfFrames;
    
    mov(1:nFrames) = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),'colormap',[]);

    k=1;
    for k = 1:nFrames
        mov(k).cdata = read(vid,k);
    end

    hf = figure;
    set(hf, 'position', [150 150 vidWidth vidHeight])

    movie(hf, mov, 1, vid.FrameRate);


 % Converts and affinity matrix to a distance matrix
function G = getDistanceMat(A)
        N = size(A,1);
        b = diag(A);
        G = b * ones(N,1)' + ones(N,1) * b' - 2 * A;
        

%This is from Blake Shaw's mvedriver.m
%Devin added polynomial kernel
function [A] = calculateAffinityMatrix(X, affType, sigma, d)
    [D,N] = size(X);
    disp(sprintf('Calculating Distance Matrix'));
    
    A = zeros(N, N);
    
    if affType == 1
        disp(sprintf('Using Linear Kernel'));
        A = X' * X; 
        %A = cov(X);
    elseif affType == 2
        disp(sprintf('Using RBF Kernel'));
        A = zeros(N, N);
        R1 = X' * X;
        Z = diag(R1);
        R2 = ones(N, 1) * Z';
        R3 = Z * ones(N, 1)';
        A  = exp((1/sigma) * R1 - (1/(2*sigma)) * R2 - (1/(2*sigma)) * R3);
    elseif affType == 3
        disp(sprintf('Using Polynomial Kernel'));
        A = (X' * X)^d;
    end
    
    

% Finds nearest neighbors in distance matrix G
% bval is number of neighbors
function neighbors = calculateNeighborMatrix(G, bVal, type)

    N=length(G);
        
    if type==1
        disp(sprintf('Finding neighbors using K-nearest -- k=%d', bVal));
        [sorted,index] = sort(G);
        nearestNeighbors = index(2:(bVal+1),:);
        
        
        neighbors = zeros(N, N);
        for i=1:N
            for j=1:bVal
                neighbors(i, nearestNeighbors(j, i)) = 1;
                neighbors(nearestNeighbors(j, i), i) = 1;
            end
        end
        
    else
        disp(sprintf('Finding neighbors using B-matching -- b=%d', bVal));
        neighbors = permutationalBMatch(G, bVal);
        neighbors = neighbors .* (1 - eye(N));
    end
    
% Plots a 2d embedding
% from the example
% Modified by Devin Jones
function plotEmbedding(Y, neighbors, plotTitle, figureNum)
    plot = figure(figureNum);
    clf;
    
    N = length(neighbors);
    
    scatter(Y(1,:),Y(2,:), 60,'filled'); axis equal;
    for i=1:N
        for j=1:N
            if neighbors(i, j) == 1
                line( [Y(1, i), Y(1, j)], [ Y(2, i), Y(2, j)], 'Color', [0, 0, 1], 'LineWidth', 1);
            end
        end
    end
    
    title(plotTitle);
    drawnow; 
    axis off;
    fileName = strcat(strrep(strrep(plotTitle,' ',''),':',''),'.png');
    saveas(plot,fileName,'png');

% Plots fidelity of MVE over various nearest neighbor params
function plotFidelity(fidelity, plotTitle, figureNum)
    plot = figure(figureNum);
    clf;
    
    bar(fidelity(:,1),fidelity(:,2));
    
    title(plotTitle);
    xlabel('Number of Nearest Neighbors');
    ylabel('Fidelity');
    drawnow; 
    
    fileName = strcat(strrep(strrep(plotTitle,' ',''),':',''),'.png');
    saveas(plot,fileName,'png');
