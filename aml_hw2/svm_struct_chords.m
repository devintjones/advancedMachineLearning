function svm_struct_chords

  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
   
  song = strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(1,:));
  load(song)
  F = F;
  L = L;
  [height,width] = size(F);
  
  
  % 30% random sample from song added to training data
  num_train = width*.3;
  training_idx = randsample(width,width*.3);
  training_idx = 1:width; % test with all frames from one song

  linear_chroma = @(F,i) F(:,i);

  % iterate over range of C margin values
  make_c_vals = @(i) 10.^(i);
  C_vals = make_c_vals(-5:5);
  
  cv_score = zeros(length(C_vals),3);
  for i=1:length(C_vals)
      args = strcat(' -c ',C_vals(i),' -o 1 -v 1 ');
      scores = svm_struct_cv(5,training_idx,F,L,args,linear_chroma);
      cv_score(i,:) = [C_val(i),mean(scores),std(scores)];
  end
 
end

  
% parallel cross validation
% parallel implementation cuts training time in half
function scores = svm_struct_cv(folds,training_idx,F,L,args,feature_maker)
  len = length(training_idx);
  chunk_size = len/folds;
  scores = zeros(folds,1);
  parfor k = 1:5
    
      % select kth subset of training_idx
      % and corresponding test set
      cv_test  = training_idx((k-1)*chunk_size + 1:k*chunk_size);
      cv_train = setdiff(training_idx,cv_test);
      
      % have to init cells in the body of parfor
      patterns_train = cell(1,length(cv_train));
      labels_train   = cell(1,length(cv_train));
      patterns_test  = cell(1,length(cv_test));
      labels_test    = cell(1,length(cv_test));
      
      for i=1:length(cv_train)
        patterns_train{i}  = feature_maker(F,cv_train(i));
        labels_train{i}    = L(cv_train(i)) ; 
      end
      for i=1:length(cv_test)
        patterns_test{i}   = feature_maker(F,cv_test(i));
        labels_test{i}    = L(cv_test(i)) ; 
      end

      % have to init parm in the body of parfor
      parm = struct
      parm.lossFn = @lossCB ;
      parm.constraintFn  = @constraintCB ;
      parm.featureFn = @featureCB ;
      parm.verbose = 0 ;

      psi = featureCB(parm, F(:,1), L(1));
      parm.dimension = length(psi);

      parm.patterns = patterns_train ;
      parm.labels = labels_train ;
  
      model = svm_struct_learn(args, parm) ;
      w = model.w ;

      % compute accuracy & store in array
      acc = accuracy(parm,w,patterns_test,labels_test);
      scores(k,1) = acc;
  end
  
end



function acc = accuracy(param,w,test_x,test_y)
  num_correct = 0;
  for i=1:length(test_x)
      prediction = predict(param,w,test_x{i});
      if prediction == test_y{i}
          num_correct = num_correct + 1;
      end
  end
  acc = num_correct/length(test_x);
end

% predict class of x based on w
function prediction = predict(param,w,x)
    for i = 0:24
      psi   = param.featureFn(param,x,i);
      score = dot(psi,w) ; 
      if ~exist('max_score')
          max_score = score;
          prediction = i;
      end
      if score > max_score
          max_score = score;
          prediction = i;
      end
    end
end 

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
  delta = double(y ~= ybar) ;
  if param.verbose
    fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
  end
end

function psi = featureCB(param, x, y)
  width = length(x);
  psi   = zeros(1,25*width);
  psi(:,(y)*width+1:(y+1)*width)= x ;
  psi = sparse(psi);
  if param.verbose
    fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
            x, y, full(psi(1)), full(psi(2))) ;
  end
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>

  % slack rescaling
  psi   = param.featureFn(param,x,y);
  for i = 0:24
      if i == y
          continue;
      end
      psi_compare   = param.featureFn(param,x,i);
      delta = param.lossFn(param, y, i);
      score = delta*(1 + dot(psi_compare,model.w) - dot(psi,model.w)) ; 
      if ~exist('max_score')
          max_score = score;
          yhat = i;
      end
      if score > max_score
          max_score = score;
          yhat = i;
      end
  end
      
  if param.verbose
    fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
            model.w, x, y, yhat) ;
  end
end
