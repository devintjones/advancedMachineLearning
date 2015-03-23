function svm_struct_chords_plot

  results = ls('*_results.mat');
  features = [];
  results_vec  = [];
  for j=1:length(results(:,1))
      features= [features,strrep(results(j,:),'_results.mat','')];
      load(results(j,:))
      results_vec = [results_vec,acc];
      disp([results(j,:),num2str(acc,'%f')])
      %disp(acc)
  end
  best_model = find(results_vec==max(results_vec));
  model_name = strrep(results(best_model,:),'_results.mat','');
  disp(['Best model: ',model_name])
  load(results(best_model,:))
  
  
  best_song_idx = [9 2];
  
  for j = 1:2
    [fig,acc]  = score_and_plot(model_name,best_song_idx(j),w,obs_subset,feature_maker);
    acc
    fig
  end
  
end

function [fig,acc] = score_and_plot(model_name,best_song_idx,w,obs_subset,feature_maker)
  
  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
  load(strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(best_song_idx,:)))
  
  plot_title = strrep(strcat(model_name,' ', songfiles(best_song_idx,:)),'_',' ');
  disp(plot_title)
    
  [height,width] = size(F);
  song_indices=[1;1+width];
  final_trains  = obs_subset(1:width,song_indices);

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

% hard to distinguish chords from features
function fig = overlayed_chart(pred_vec,actual_vec,final_trains,F)
  fig = figure;  
  [ax,h1,h2] = plotyy(final_trains,[pred_vec+10;actual_vec+10],final_trains,F(:,final_trains));
  legend('predicted','actual')
  title(strrep(strcat(strrep(results(best_model,:),'_results.mat',''),'-', songfiles(best_song,:)),'_',' '))
  xlabel('Frame #')

  ylabel(ax(1),'Actual/Predicted') % left y-axis
  ylabel(ax(2),'Chroma Feature Values') % right y-axis
  h1(1).LineWidth=1.5;
  h1(1).LineStyle='-.';
  h1(2).LineWidth=1.5;
  h1(2).LineStyle='--';
end


function best_song = find_best_song(results,best_model)
  % find best performing songs using the best performing model
  load(results(best_model,:))
  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
  randn('state',0) ;
  rand('state',0) ;
  random_songs = randsample(length(songfiles),10);
  
  song_acc = [];
  for i=1:length(random_songs)
      song = strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(random_songs(i),:));
      load(song)
      
      [height,width] = size(F);
      song_indices=[1;1+width];
      final_trains  = obs_subset(1:width,song_indices);

      for j=1:length(final_trains)
        patterns_test{j}  = feature_maker(F,final_trains(j));
        labels_test{j}    = L(j) ; 
      end
      
      parm.featureFn = @featureCB;
      parm.verbose = 0;
      
      acc = accuracy(parm,w,patterns_test,labels_test);
      song_acc = [song_acc;acc];
  end
  
  songs = [transpose(1:length(song_acc)),song_acc];
  sorted_songs = sortrows(songs,-2);
  best_song = sorted_songs(1:2,1);
  disp('Best song:')
  best_song
end

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
