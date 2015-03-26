function svm_hmm

  randn('state',0) ;
  rand('state',0) ;

  % read data and pick random subset of songs
  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
  random_songs = randsample(length(songfiles),10);


  % iterate over each feature that are compared in the analysis
  feature_list = {@linear_chroma,@linear_chroma_quad,...
                  @one_before,@two_before,@three_before,...
                  @one_after, @two_after, @three_after,...
                  @one_before_after,@two_before_after,@three_before_after,...
                  @one_after_quad, @two_after_quad, @three_after_quad,...
                  @one_before_after_quad,@two_before_after_quad,@three_before_after_quad};
  for j=1:length(feature_list)
      
      % select feature maker
      [feature_maker,obs_subset,filename] = feature_list{j}();
      disp(['Fitting feature: ',strrep(filename,'_results.mat','')])
      filename = ['hmm_', filename];
      
      % put relevant data in key value structure
      % to aggregate stats over the song results for each feature
      song_scores = containers.Map();
      c_final     = containers.Map();
      w_final     = containers.Map();
      for i=1 %:length(random_songs)
          song = get_song_name(songfiles,random_songs(i));
          load(song)
          % parfor complains if these are not defined explicitly
          F=F;
          L=L;

          % keep track of song indices to make sure we don't build features
          % that straddle songs
          [height,width] = size(F);
          song_indices = [1;1+width]; 

          % 30% random sample from song added to training data
          training_idx = randsample(width,round(width*.3));
          final_trains = obs_subset(training_idx,song_indices); % some frames need to be removed because of certain feature builders
          
          % k fold cross validation to find best C
          make_c_vals = @(k) 10.^(k);
          tune_val = ' -c ';
          tuning_param = make_c_vals(-2:2);
          
          cv_score = zeros(length(tuning_param),3);
          for c=1:length(tuning_param)
              args = [' -c ',num2str(tuning_param(c),'%f'),' -e .1 -v 0 '];
              disp(['Fitting model: ',args])
              scores = svm_hmm_cv(3,final_trains,F,L,args,feature_maker);
              scores
              cv_score(c,:) = [tuning_param(c),mean(scores),std(scores)];
          end
          cv_score
          
          
          % select best Cval from cross validation
          c      = find(cv_score(:,2)==max(cv_score(:,2)),1);
          best_c = tuning_param(c);
          args   = [tune_val,num2str(best_c,'%f'),' -e .1 -v 0 '];
          disp(['Using ',tune_val,' = ',num2str(best_c,'%f'),' for testing parameters'])

          % retrain using best c on original %30
          % saves the model file for prediction later
          params_train  = make_data(final_trains,feature_maker,F,L);
          model_filename= strcat(strrep(songfiles(random_songs(i),:),'.mat',''),strrep(filename,'.mat','.chords'));
          success       = svm_hmm_learn(args,params_train,model_filename); 
          
          % 10% testing data
          unusedF      = setdiff(1:width,training_idx);
          test_idx     = randsample(unusedF,round(.1*width));
          test_idx     = obs_subset(test_idx,song_indices);
          params_test  = make_data(test_idx,feature_maker,F,L);
          
          % compute accuracy & store in array
          acc = score_model(params_test  ,model_filename);
          sprintf('Final Accuracy: %f',acc)

          % log important data for aggregation/reference
          song_scores(songfiles(random_songs(i),:)) = acc ;
          c_final(songfiles(random_songs(i),:))     = best_c ; 
      end
      
      % save a .mat file for each feature
      save(filename)
      
   end
  
end

% k fold cross validation
function scores = svm_hmm_cv(folds,training_idx,F,L,args,feature_maker)
  if folds < 3
      error('Enter fold value greater than 2')
  end
  
  len = length(training_idx);
  chunk_size = len/folds;
  scores = zeros(folds,1);
  
  for k = 1:folds
      % select kth subset of training_idx
      % and corresponding test set
      cv_test  = training_idx(round((k-1)*chunk_size) + 1:round(k*chunk_size));
      cv_train = setdiff(training_idx,cv_test);
      
      params_test  = make_data(cv_test, feature_maker,F,L);
      params_train = make_data(cv_train,feature_maker,F,L);
      
      success    = svm_hmm_learn(args,params_train,'modelfile.dat'); 
      
      % compute accuracy & store in array
      acc = score_model(params_test  ,'modelfile.dat');
          
      scores(k,1) = acc;
  end
  
end

% score data based on a specified model_file
function acc = score_model(parameters,model_file)
  pred_array = svm_hmm_classify(parameters,model_file);
  num_correct = 0;
  for i=1:length(pred_array)
    %disp([pred_array(i,1),parameters.labels{i}])
    if pred_array(i,1) == parameters.labels{i}
        num_correct = num_correct + 1;
    end
  end
  acc = num_correct / length(pred_array);
end
          
function song = get_song_name(songfiles,i)
  song = strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(i,:));
end

% executes the feature_maker and puts data into a standardized format
% based on data structures from the matlab api for svm_struct
function param = make_data(test_idx,feature_maker,F,L)
    for b=1:length(test_idx)
        patterns_test{b}  = feature_maker(F,test_idx(b));
        labels_test{b}    = L(test_idx(b)) + 1 ; 
    end
    param = struct;
    param.patterns = patterns_test;
    param.labels = labels_test;
end
          
          

% menu of feature_makers
% must take only F,i as arguements
% and return an array of functions and a file name

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

% added quadratic cross terms
% Q+0-0
function [f,obs_subset,file_name] = linear_chroma_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) training_idx;
  file_name  = 'linear_chroma_quad_results.mat';
end


% Q+1-0
function [f,obs_subset,file_name] = one_after_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i);F(:,i+1)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,song_indices-1);
  file_name  = 'one_after_quad_results.mat';
end

% Q+2-0
function [f,obs_subset,file_name] = two_after_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i);F(:,i+1);F(:,i+2)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices-1,song_indices-2));
  file_name  = 'two_after_quad_results.mat';
end

% Q+3-0
function [f,obs_subset,file_name] =  three_after_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i);F(:,i+1);F(:,i+2);F(:,i+3)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices-1,song_indices-2,song_indices-3));
  file_name  = 'three_after_quad_results.mat'; 
end

% Q+1-1
function [f,obs_subset,file_name] = one_before_after_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i);F(:,i-1);F(:,i+1)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , song_indices -1));
  file_name = 'one_before_after_quad_results.mat';
end

% Q+2-2
function [f,obs_subset,file_name] = two_before_after_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i);F(:,i-1);F(:,i+1);F(:,i-2);F(:,i+2)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , ...
      song_indices -1,song_indices-2,song_indices+1));
  file_name = 'two_before_after_quad_results.mat';  
end

% Q+3-3
function [f,obs_subset,file_name] = three_before_after_quad
  function feature = feature_maker(F,i) 
      feature_init = [F(:,i);F(:,i-1);F(:,i+1);F(:,i-2);F(:,i+2);F(:,i-3);F(:,i+3)];
      % quadratic cross terms will be the size of a finite arithmetic sum
      big_feature  = zeros((length(feature_init)^2+length(feature_init))/2,1);
      idx = 1;
      for i = 1:length(feature_init)
        for j = 1:length(feature_init)
          if j >= i
              big_feature(idx) = feature_init(i)*feature_init(j);
              idx = idx + 1;
          end
        end
      end
      feature = [feature_init;big_feature];
  end
  f          = @feature_maker;
  obs_subset = @(training_idx,song_indices) setdiff(training_idx,vertcat(song_indices , ...
      song_indices -1,song_indices-2,song_indices+1,song_indices-3,song_indices+2));
  file_name = 'three_before_after_quad_results.mat';  
end


% trains svm_hmm
function ret_val = svm_hmm_learn(args, param, modelfile)
  % write patterns to .dat file
  
  % put data in big cell array
  feature_dim = length(param.patterns{1});
  data = cell(length(param.patterns),feature_dim +1);
  for i=1:length(param.patterns)
      data(i,1) = num2cell(param.labels{i});
      for j=1:feature_dim
        data(i,j+1) = {[num2str(j),':',num2str(param.patterns{i}(j,1))]};
      end
  end
  
  % write cell array to .dat file
  fileID = fopen('training.dat','w');
  formatSpec = '%d';
  for i=1:feature_dim
      formatSpec = strcat(formatSpec,' %s');
  end
  formatSpec = strcat(formatSpec,'\n');
  
  for i=1:length(param.patterns)
      fprintf(fileID,formatSpec,data{i,:});
  end
  fclose(fileID);  

  % fit model
  ret_val = system(['svm_hmm_windows\svm_hmm_learn',args, 'training.dat ',modelfile]);
  
end

% predicts chord for a dataset based on modelfile
function pred_array = svm_hmm_classify(param, modelfile)
% put data in big cell array
  feature_dim = length(param.patterns{1});
  data = cell(length(param.patterns),feature_dim +1);
  for i=1:length(param.patterns)
      data(i,1) = num2cell(param.labels{i});
      for j=1:feature_dim
        data(i,j+1) = {[num2str(j),':',num2str(param.patterns{i}(j,1))]};
      end
  end
  
  % write cell array to .dat file
  fileID = fopen('scoring.dat','w');
  formatSpec = '%d';
  for i=1:feature_dim
      formatSpec = strcat(formatSpec,' %s');
  end
  formatSpec = strcat(formatSpec,'\n');
  
  for i=1:length(param.patterns)
      fprintf(fileID,formatSpec,data{i,:});
  end
  fclose(fileID);  

  % score the file
  ret_val = system(['svm_hmm_windows\svm_hmm_classify scoring.dat ', modelfile,' results.chords']);
  
  % read results back into matlab
  fileID = fopen('results.chords');
  prediction = textscan(fileID,'%d');
  fclose(fileID);
  
  pred_array = prediction{1};
end
    
