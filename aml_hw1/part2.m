
% Advacned Machine Learning HW1
% Professor Toney Jebera, Columbia University
% Author: Devin Jones
function part2
run('C:\Users\Devin\Documents\MATLAB\mve-05\setuppath.m');

load('digits9.mat');

%load digits9.mat; for i=1:155; plot(digits9(i,1:2:140),digits9(i,2:2:140),'.'); pause(0.4); end
X = digits9;
[D, N] = size(X);
disp(sprintf('%d points in %d dimensions:', N, D));

% Calculate linear kernal. 
A = calculateAffinityMatrix(X,3,0,4);

% Derive distance matrix from kernal
G = convertAffinityToDistance(A);


% Neighbor params
bVal = 3; %# of neighbors
type = 1; %KNN

% Returns a binary matrix with the dimensions of G
% 1 is a neighbor, 0 is not a neighbor
neighbors = calculateNeighborMatrix(G, bVal, type);

%parameters for MVE
targetd = 2;
tol = 0.99;

[Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);

disp(sprintf('The size of Y is %d,%d',size(Y)));
plotEmbedding(Y, neighbors, 'MVE embedding' ,35)

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
        disp(sprintf('Using Polynomical Kernel'));
        A = (X' * X)^d;
    end
    
 % Converts and affinity matrix to a distance matrix
 % Modified by Devin Jones
function G = convertAffinityToDistance(A)
        N = size(A,1);
        b = diag(A);
        G = b * ones(N,1)' + ones(N,1) * b' - 2 * A;
    

% Finds nearest neighbors in distance matrix G
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
function plotEmbedding(Y, neighbors, plotTitle, figureNum)
    figure(figureNum);
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