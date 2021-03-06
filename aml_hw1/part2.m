
% Advanced Machine Learning HW1
% Professor Tony Jebara, Columbia University
% Author: Devin Jones
function part2

%need to run this first to ensure dependencies are included in path vars
%run('C:\Users\Devin\Documents\MATLAB\mve-05\setuppath.m')

%load digits9.mat; for i=1:155; plot(digits9(i,1:2:140),digits9(i,2:2:140),'.'); pause(0.4); end
load('digits9.mat');

X = convertDigits(digits9,0);

[D, N] = size(X);
disp(sprintf('%d points in %d dimensions:', N, D));


for kernel = 1:3; % iterate over each kernel type
    if kernel == 1
        plotTitle = 'Linear Kernel';
        sigma = 0;
        degree = 0;
    elseif kernel == 2
        sigma = .5;
        degree = 0;
        plotTitle = sprintf('RBF Kernel. Sigma = %d',sigma);
    elseif kernel== 3
        sigma = 0;
        degree = 2;
        plotTitle = sprintf('Polynomial Kernel. Degree = %d',degree);
    end;
    
    % Calculate linear kernal. 
    A = calculateAffinityMatrix(X,kernel,sigma,degree);

    % Derive distance matrix from kernal
    G = convertAffinityToDistance(A);

    % Neighbor params
    bVal = 3; %# of neighbors
    type = 1; %KNN

    % Returns a neighborhood graph with dimensions of G
    neighbors = calculateNeighborMatrix(G, bVal, type);

    %parameters for MVE
    targetd = 2;
    tol = 0.99;

    [Y, K, eigVals, mveScore] = mve(A, neighbors, tol, targetd);

    plotEmbedding(Y, neighbors, plotTitle ,kernel)
end
    
%Converts the digits9 dataset into BW image format
function [new_x] = convertDigits(digits9,movie)

    X = digits9;

    %Get dimensions of images. Its index begins at zero, so add one to comply
    %with matlab indexing
    max_dim = max(max(X)) + 1;

    [rows,cols] = size(X);

    % This will house all images of digits in its rows
    new_x = zeros(rows,max_dim^2);

    for i = 1:rows;
        %This will populate with pixels from digit9
        empty_mat = zeros(max_dim,max_dim);
        for j=1:2:cols
            y_val = max_dim - X(i,j+1);
            x_val = X(i,j) +1;
            empty_mat(y_val,x_val) = 1;
        end
        %disp(sprintf('Size of empty_mat: %d by %d',size(empty_mat)));
        new_x(i,:) = reshape(empty_mat,1,max_dim^2);
    end
    
    % Check format of the first frame
    first_frame = reshape(new_x(1,:),max_dim,max_dim);
    
    % Watch movie of frames
    if movie == 1
        for i=1:rows; imshow(reshape(new_x(i,:),max_dim,max_dim)); pause(0.4); end;
    end

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
