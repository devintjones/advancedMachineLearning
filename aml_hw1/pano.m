
% Advanced Machine Learning HW1
% Professor Tony Jebara, Columbia University
% Author: Devin Jones
%
% I manually edited this file to create frame sequences of each movie
function pano

%load digits9.mat; for i=1:155; plot(digits9(i,1:2:140),digits9(i,2:2:140),'.'); pause(0.4); end
load('digits9.mat');

X = convertDigits(digits9,0);

max_dim = max(max(digits9)) + 1

[rows,cols] = size(digits9);

pano = reshape(X(1,:),max_dim,max_dim);

for i=15:14:140
    frame_new = reshape(X(i,:),max_dim,max_dim);
    pano = horzcat(pano,frame_new);
end

imshow(pano)

imwrite(pano,'digits9.png')


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
