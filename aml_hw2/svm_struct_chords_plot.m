function svm_struct_chords_plot
  
  [best_song,best_model] = get_feature_scores();
  
  for i=1:2
      fig = plot_chroma(best_model{i},best_song{i}{1})
  end

end

% this does two things
% 1) reads all results files and compiles stats for each model and saves to
% a csv for presentation
% 2) finds the two best scoring songs to plot and information to retrieve
% their data for scoring
function [best_song,model_select] = get_feature_scores()
  
  feature_result_files = ls('*_songs_*_results.mat');
  max_score      = [0 0];
  feature_scores = table();
  
  for a=1:length(feature_result_files(:,1))
    load(feature_result_files(a,:))
    
    % meta data
    feature = strrep(strrep(feature_result_files(a,:),'_results.mat',''),'_songs','');
    
    % aggregate stats for model performance over the songs
    vals = values(song_scores);
    avg_score = mean([vals{:}]);
    std_score = std( [vals{:}]);
  
    % store in a table
    feature_scores = [feature_scores;table({feature},avg_score,std_score)];
    
    % check if best score
    if max([vals{:}]) > max_score(1)
        max_score(1) = max([vals{:}]);
        idx          = find(max([vals{:}])==[vals{:}]);
        song_names   = keys(song_scores);
        best_song{1} = song_names(idx);
        model_select{1} = feature_result_files(a,:);
    % check if second best score
    elseif max([vals{:}]) > max_score(2)
        max_score(2) = max([vals{:}]);
        idx          = find(max([vals{:}])==[vals{:}]);
        song_names   = keys(song_scores);
        best_song{2} = song_names(idx);
        model_select{2} = feature_result_files(a,:);
    end       
  end
  
  writetable(feature_scores,'feature_scores.csv')
end

% loads data, predicts chord, gets accuracy, and plots
function fig = plot_chroma(model_select,best_song)
  load(model_select)
  w = w_final(best_song);   
  
  % get feature maker
  feature_list = {@linear_chroma,@linear_chroma_quad...
                @one_before,@two_before,@three_before,...
                @one_after, @two_after, @three_after,...
                @one_before_after,@two_before_after,@three_before_after,...
                @one_after_quad, @two_after_quad, @three_after_quad,...
                @one_before_after_quad,@two_before_after_quad,@three_before_after_quad,...
                };
  feature_names = {'linear_chroma','linear_chroma_quad',...
                'one_before','two_before','three_before',...
                'one_after', 'two_after', 'three_after',...
                'one_before_after','two_before_after','three_before_after',...
                'one_after_quad', 'two_after_quad', 'three_after_quad',...
                'one_before_after_quad','two_before_after_quad','three_before_after_quad'};
  feature_map = containers.Map(feature_names,feature_list);
  
  split = strsplit(model_select,'_');
  feature_maker = strjoin(split(1,3:length(split)-1),'_');
  
  feature_select = feature_map(feature_maker);
  [feature_maker,obs_subset,filename] = feature_select();
  file = strcat('advancedMachineLearning\aml_hw2\CHORDS\',best_song);
  
  load(file)
  
  plot_title = strrep(strrep(strcat(model_select,' ',best_song),'_',' '),'.mat',' ');
  
  [height,width] = size(F);
  song_indices=[1;1+width];
  final_trains  = obs_subset(1:width,song_indices);

  % this index is a lower case L. 
  for l=1:length(final_trains)
    patterns_test{l}  = feature_maker(F,final_trains(l));
    labels_test{l}    = L(final_trains(l)) ; 
  end

  parm.featureFn = @featureCB;
  parm.verbose = 0;
  [acc,pred_vec,actual_vec] = accuracy(parm,w,patterns_test,labels_test);
  
  % plot stuff
  % http://stackoverflow.com/questions/11757987/matlab-stacking-of-various-plots
  fig = figure;
  h(1) = subplot(2,1,1); % upper plot
  plot(final_trains, [pred_vec;actual_vec]);
  set(h(1),'xticklabel',[]);
  legend('predicted','actual')
  xlim([0 max(final_trains)])
  ylabel('Chord Actual & Predicted')
  title(plot_title)
  
  h(2) = subplot(2,1,2); % lower plot
  plot(final_trains, F(:,final_trains));
  xlim([0 max(final_trains)])
  ylabel('Log Chroma Features')
  
  acc_string = sprintf('Accuracy: %f',acc);
  text(max(final_trains),max(max(F(:,final_trains))),acc_string,'HorizontalAlignment','right');
  
  linkaxes(h,'x');
  xlabel('Chroma Frame')
  
  pos=get(h,'position');
  bottom=pos{2}(2);
  top=pos{1}(2)+pos{1}(4);
  plotspace=top-bottom;
  pos{2}(4)=plotspace/2;
  pos{1}(4)=plotspace/2;
  pos{1}(2)=bottom+plotspace/2;
  set(h(1),'position',pos{1});
  set(h(2),'position',pos{2});
end

%==============================
% same prediction and evalution methods from the training program
%===========================

function [acc,pred_vec,actual_vec] = accuracy(param,w,test_x,test_y)
  num_correct = 0;
  pred_vec    = [];
  actual_vec  = [];
  for i=1:length(test_x)
      prediction = predict(param,w,test_x{i});
      pred_vec = [pred_vec,prediction];
      actual_vec = [actual_vec,test_y{i}];
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
      %disp(score)
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

%========================
% same feature makers from the training program
%========================

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

