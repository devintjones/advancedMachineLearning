function svm_hmm_plot
  
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
  
  feature_result_files = ls('*hmm_*_results.mat');
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
  
  writetable(feature_scores,'hmm_feature_scores.csv')
end

          
function param = make_data(test_idx,feature_maker,F,L)
    for b=1:length(test_idx)
        patterns_test{b}  = feature_maker(F,test_idx(b));
        labels_test{b}    = L(test_idx(b)) + 1 ; 
    end
    param = struct;
    param.patterns = patterns_test;
    param.labels = labels_test;
end



% loads data, predicts chord, gets accuracy, and plots
function fig = plot_chroma(model_select,best_song)

          
          
  % lookup feature_maker from string
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
  
  
  % fetch feature params
  split = strsplit(model_select,'_');
  feature_maker = strjoin(split(1,2:length(split)-1),'_');
  feature_select = feature_map(feature_maker);
  [feature_maker,obs_subset,filename] = feature_select();
  
  % get raw data and massage for prediction & plot
  song_data_file = strcat('advancedMachineLearning\aml_hw2\CHORDS\',best_song);
  load(song_data_file)
  
  [height,width] = size(F);
  song_indices   =[1;1+width];
  final_trains   = obs_subset(1:width,song_indices);
  
  % predict chords
  params_test    = make_data(final_trains  ,feature_maker,F,L);
  model_filename = strcat(strrep(best_song,'.mat',''),strrep(model_select,'.mat','.chords'));
  [acc, plot_agreement]= score_model(params_test  ,model_filename);
  
  size(plot_agreement)
  size(final_trains)
  
  plot_title = strrep(strrep(strcat(model_select,' ',best_song),'_',' '),'.mat',' ');
  
  % plot stuff
  % http://stackoverflow.com/questions/11757987/matlab-stacking-of-various-plots
  fig = figure;
  h(1) = subplot(2,1,1); % upper plot
  plot(final_trains, plot_agreement);
  set(h(1),'xticklabel',[]);
  legend('predicted','actual')
  xlim([0 max(final_trains)])
  ylim([1 25])
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

function [acc,plot_agreement] = score_model(parameters,model_file)
  pred_array = svm_hmm_classify(parameters,model_file);
  num_correct = 0;
  plot_agreement = [];
  for i=1:length(pred_array)
    plot_agreement = [plot_agreement; [pred_array(i,1),parameters.labels{i}]];
    %disp([pred_array(i,1),parameters.labels{i}])
    if pred_array(i,1) == parameters.labels{i}
        num_correct = num_correct + 1;
    end
  end
  plot_agreement = transpose(plot_agreement);
  acc            = num_correct / length(pred_array);
end


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

%========================
% same feature makers from the training program
% didn't want to save ~20 matlab files to import externally
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

