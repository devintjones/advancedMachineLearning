function svm_struct_chords

  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
   
  song = strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(1,:));
  load(song)

  [height,width] = size(F);
  
  % build joint feature map
  % and corresponding y vector
  patterns = {} ;
  labels = {} ;
  for i=1:width
    patterns{i}          = F(:,i);
    labels{i}            = L(i) ; 
  end

  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.lossFn = @lossCB ;
  parm.constraintFn  = @constraintCB ;
  parm.featureFn = @featureCB ;
  parm.verbose = 1 ;
  
  test_x = patterns{1};
  test_y = labels{1};
  psi = featureCB(parm, test_x, test_y);
  parm.dimension = length(psi);

  args = ' -c 1.0 -o 1 -v 1 ';
  model = svm_struct_learn(args, parm) ;
  w = model.w ;

  %{
  % ------------------------------------------------------------------
  %                                                              Plots
  % ------------------------------------------------------------------
  
  figure(1) ; clf ; hold on ;
  x = [patterns{:}] ;
  y = [labels{:}] ;
  plot(x(1, y>0), x(2,y>0), 'g.') ;
  plot(x(1, y<0), x(2,y<0), 'r.') ;
  set(line([0 w(1)], [0 w(2)]), 'color', 'y', 'linewidth', 4) ;
  xlim([-3 3]) ;
  ylim([-3 3]) ;
  set(line(10*[w(2) -w(2)], 10*[-w(1) w(1)]), ...
      'color', 'y', 'linewidth', 2, 'linestyle', '-') ;
  axis equal ;
  set(gca, 'color', 'b') ;
  w
  %}
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
  
  psi   = param.featureFn(param,x,y);
  size(psi)
  size(model.w)
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
