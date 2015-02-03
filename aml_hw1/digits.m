function digits

load('digits9.mat');

X = digits9;

%Get dimensions of images. It index begins at zero, so add one to comply
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
first_frame = reshape(new_x(1,:),max_dim,max_dim);

for i=1:rows; imshow(reshape(new_x(i,:),max_dim,max_dim)); pause(0.4); end;
