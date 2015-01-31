
% Advacned Machine Learning HW1
% Professor Toney Jebera, Columbia University
% Author: Devin Jones
function part1
%run('C:\Users\Devin\Documents\MATLAB\mve-05\setuppath.m');

% Load data and visualize spinning teapot
load teapots100.mat; for i=1:100; image(reshape(teapots(:,i),76,101,3)/255); pause(0.01); end;

X = teapots;
[D, N] = size(X);
disp(sprintf('%d points in %d dimensions:', N, D));

% Calculate linear kernal. 
A = X' * X; 
    

% Derive distance matrix from kernal
G = getDistanceMat(A);


% Iterate over 2-4 nearest neighbors
for i=2:4
    mve2d(G,i,A);
end

% Iterate 2D MVE embedding over number of neighbors
function mve2d(G,bVal,A)
    
    % Returns a binary matrix with the dimensions of G
    % 1 is a neighbor, 0 is not a neighbor
    neighbors = nearestNeighbors(G, bVal);

    %parameters for MVE
    targetd = 2;
    tol = 0.99;

    [Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);

    plotTitle = sprintf('MVE embedding: Teapot images. Number of neighbors: %d',bVal);
    plotEmbedding(Y, neighbors, plotTitle ,bVal)
    
 % Converts and affinity matrix to a distance matrix
function G = getDistanceMat(A)
        N = size(A,1);
        b = diag(A);
        G = b * ones(N,1)' + ones(N,1) * b' - 2 * A;
    

% Finds nearest neighbors in distance matrix G
% From the example
% Modified by Devin Jones
function neighbors = nearestNeighbors(G, bVal)

    N=length(G);
        
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
    
    
    