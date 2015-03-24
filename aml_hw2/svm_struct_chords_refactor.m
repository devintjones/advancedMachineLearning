function svm_struct_chords_refactor

  randn('state',0) ;
  rand('state',0) ;

  % read data and pick random subset of songs
  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
  random_songs = randsample(length(songfiles),10);


  % iterate over each feature that are compared in the analysis
  feature_list = {@linear_chroma,...
                  @one_before,@two_before,@three_before,...
                  @one_after, @two_after, @three_after,...
                  @one_before_after,@two_before_after,@three_before_after};
  for j=1:length(feature_list)
      
      % select feature maker
      [feature_maker,obs_subset,filename] = feature_list{j}();
      disp(['Fitting feature: ',strrep(filename,'_results.mat','')])
      filename = ['songs_', filename];
      
      % put relevant data in key value structure
      % to aggregate stats over the song results for each feature
      song_scores = containers.Map();
      c_final     = containers.Map();
      w_final     = containers.Map();
      for i=1:length(random_songs)
          song = get_song_name(songfiles,random_songs(i));
          load(song)
          
          % parfor complains if these are not redefined
          F=F;
          L=L;

          % keep track of song indices to make sure we don't build features
          % that straddle songs
          [height,width] = size(F);
          song_indices = [1;1+width]; 


          % 30% random sample from song added to training data
          training_idx = randsample(width,round(width*.3));
          final_trains  = obs_subset(training_idx,song_indices); % some frames need to be removed because of certain feature builders

          
          % svm_struct args
          make_c_vals = @(k) 10.^(k);
          C_vals = make_c_vals(-3:1);

          constraint_param = ' -o 2 ';
          verbosity = ' -v 0 ';

          cv_score = zeros(length(C_vals),3);
          for m=1:length(C_vals)
              args = [' -c ',num2str(C_vals(m),'%f'),constraint_param,verbosity];
              disp(['Fitting model: ',args])
              scores = svm_struct_cv(3,final_trains,F,L,args,feature_maker);
              scores 
              cv_score(m,:) = [C_vals(m),mean(scores),std(scores)];
          end
          cv_score

          
          % select best Cval from cross validation
          c      = find(cv_score(:,2)==max(cv_score(:,2)),1);
          best_c = C_vals(c);
          args   = [' -c ',num2str(best_c,'%f'),constraint_param,verbosity];
          disp(['Using C = ',num2str(best_c,'%f'),' for final parameters'])

          
          % finally, fit to entire training data with the best C parameter and 
          % measure accuracy on unseen test data (10%)
          unusedF   = setdiff(1:width,training_idx);
          test_idx  = randsample(unusedF,round(.1*width));
          test_idx  = obs_subset(test_idx,song_indices);

          [w,acc] = train_svm_struct(final_trains,test_idx,F,L,args,feature_maker);
          sprintf('Final Accuracy: %f',acc)
          
          % log import data for aggregation
          song_scores(songfiles(random_songs(i),:)) = acc ;
          c_final(songfiles(random_songs(i),:))     = best_c ; 
          w_final(songfiles(random_songs(i),:))     = w ;
      end
      
      % keep track of which constraint I used
      if findstr(args,'-o 1')
        filename = ['slack_',filename];
      else
        filename = ['margin_',filename];  
      end

      % so that we can read song_scores later and compare across features
      save(filename)
  end
  
end

function song = get_song_name(songfiles,i)
  song = strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(i,:));
end
          

% menu of feature_makers
% must take only F,i as arguements
% and return a vector

% +0-0
function [f,obs_subset,file_name] = linear_chroma
  f = @(F,i) [F(:,i)];
  obs_subset = @(training_idx,song_indices) training_idx;
  file_name = 'linear_chroma_results.mat';
end

% +0-1
function [f,obs_subset,file_name] = one_before
  f = @(F,i) [F(:,i);F(:,i-1)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,song_indices);
  file_name = 'one_before_results.mat';
end

% +0-2
function [f,obs_subset,file_name] = two_before
  f = @(F,i) [F(:,i);F(:,i-1);F(:,i-2)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , song_indices + 1));
  file_name = 'two_before_results.mat';  
end

% +0-3
function [f,obs_subset,file_name] = three_before
  f = @(F,i) [F(:,i);F(:,i-1);F(:,i-2);F(:,i-3)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , song_indices + 1,song_indices + 2));
  file_name = 'three_before_results.mat';
end

% +1-0
function [f,obs_subset,file_name] = one_after
  f = @(F,i) [F(:,i);F(:,i+1)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,song_indices-1);
  file_name = 'one_after_results.mat';
end

% +2-0
function [f,obs_subset,file_name] = two_after
  f = @(F,i) [F(:,i);F(:,i+1);F(:,i+2)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices-1,song_indices-2));
  file_name = 'two_after_results.mat';
end

% +3-0
function [f,obs_subset,file_name] =  three_after
  f = @(F,i) [F(:,i);F(:,i+1);F(:,i+2);F(:,i+3)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices-1,song_indices-2,song_indices-3));
  file_name = 'three_after_results.mat'; 
end

% +1-1
function [f,obs_subset,file_name] = one_before_after
  f = @(F,i) [F(:,i);F(:,i-1);F(:,i+1)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , song_indices -1));
  file_name = 'one_before_after_results.mat';
end

% +2-2
function [f,obs_subset,file_name] = two_before_after
  f = @(F,i) [F(:,i);F(:,i-1);F(:,i+1);F(:,i-2);F(:,i+2)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , ...
      song_indices -1,song_indices-2,song_indices+1));
  file_name = 'two_before_after_results.mat';  
end

% +3-3
function [f,obs_subset,file_name] = three_before_after
  f = @(F,i) [F(:,i);F(:,i-1);F(:,i+1);F(:,i-2);F(:,i+2);F(:,i-3);F(:,i+3)];
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , ...
      song_indices -1,song_indices-2,song_indices+1,song_indices-3,song_indices+2));
  file_name = 'three_before_after_results.mat';  
end


% parallel cross validation
% for tuning C parameter
% parallel implementation cuts training time in half
function scores = svm_struct_cv(folds,training_idx,F,L,args,feature_maker)
  if folds < 3
      error('Enter fold value greater than 2')
  end
  
  len = length(training_idx);
  chunk_size = len/folds;
  scores = zeros(folds,1);
  
  parfor k = 1:folds
      % select kth subset of training_idx
      % and corresponding test set
      cv_test  = training_idx(round((k-1)*chunk_size) + 1:round(k*chunk_size));
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
        labels_test{i}     = L(cv_test(i)) ; 
      end

      % have to init parm in the body of parfor
      parm = struct
      parm.lossFn = @lossCB ;
      if findstr(args,'-o 1')
        parm.constraintFn  = @slackConstraintCB;
      else
        parm.constraintFn  = @marginConstraintCB;
      end
      parm.featureFn = @featureCB ;
      parm.verbose = 0 ;

      psi = featureCB(parm, feature_maker(F,10), L(10));
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

% trains and tests on specified indices of F
% returns final w
function [w,acc] = train_svm_struct(training_idx,test_idx,F,L,args,feature_maker)
      disp('Training on entire training set')
      for i=1:length(training_idx)
        patterns_train{i}  = feature_maker(F,training_idx(i));
        labels_train{i}    = L(training_idx(i)) ; 
      end
      for i=1:length(test_idx)
        patterns_test{i}  = feature_maker(F,test_idx(i));
        labels_test{i}    = L(test_idx(i)) ; 
      end
      
      parm = struct;
      parm.lossFn = @lossCB ;
      if findstr(args,'-o 1')
        parm.constraintFn  = @slackConstraintCB;
      else
        parm.constraintFn  = @marginConstraintCB;
      end
      parm.featureFn = @featureCB ;
      parm.verbose = 0 ;

      psi = featureCB(parm, feature_maker(F,10), L(10));
      parm.dimension = length(psi);

      parm.patterns = patterns_train ;
      parm.labels = labels_train ;
  
      model = svm_struct_learn(args, parm) ;
      w = model.w ;
      acc = accuracy(parm,w,patterns_test,labels_test);
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
      score = dot(transpose(w),psi) ; 
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
  psi = transpose(psi);
  if param.verbose
    fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
            x, y, full(psi(1)), full(psi(2))) ;
  end
end

function yhat = slackConstraintCB(param, model, x, y)
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
      score = delta*(1 + dot(transpose(model.w),psi_compare) - dot(transpose(model.w),psi)) ; 
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

function yhat = marginConstraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>

  % slack rescaling
  psi   = param.featureFn(param,x,y);
  for i = 0:24
      if i == y
          continue;
      end
      delta = param.lossFn(param, y, i);
      score = delta + dot(transpose(psi),model.w) ; 
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