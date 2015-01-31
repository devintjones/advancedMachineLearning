
% Advacned Machine Learning HW1
% Professor Toney Jebera, Columbia University
% Author: Devin Jones
function part1
run('C:\Users\Devin\Documents\MATLAB\mve-05\setuppath.m');

load('teapots100.mat');

%load teapots100.mat; for i=1:100; image(reshape(teapots(:,i),76,101,3)/255); pause(0.01); end;
X = teapots;
[D, N] = size(X);
disp(sprintf('%d points in %d dimensions:', N, D));

% Calculate linear kernal. 
A = calculateAffinityMatrix(X,1,0);

% Derive distance matrix from kernal
G = convertAffinityToDistance(A);


% Neighbor params
bVal = 3;
type = 1;

neighbors = calculateNeighborMatrix(G, bVal, type);


%parameters for MVE
targetd = 2;
tol = 0.99;

[Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);


%this is from Blake Shaw's mvedrive.m
function [A] = calculateAffinityMatrix(X, affType, sigma)
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
    
