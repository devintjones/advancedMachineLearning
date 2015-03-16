function svm_struct_chords

  songfiles = ls('advancedMachineLearning\aml_hw2\CHORDS\*.mat');
   
  song = strcat('advancedMachineLearning\aml_hw2\CHORDS\',songfiles(1,:));
  load(song)

  [height,width] = size(F);
  
  % build joint feature map
  % and corresponding y vector
  patterns = {} ;
  labels = {} ;
  for i=1:10
    patterns{i}          = zeros(25,height);
    patterns{i}(L(i)+1,:)= transpose(F(:,i)) ;
    labels{i}            = -1*ones(1,25) ; 
    labels{i}(L(i)+1)    = 1 ;
  end

  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.lossFn = @lossCB ;
  parm.constraintFn  = @constraintCB ;
  parm.featureFn = @featureCB ;
  parm.dimension = height ;
  parm.verbose = 1 ;
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
  delta = y - ybar ;
  if param.verbose
    fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
  end
end

function psi = featureCB(param, x, y)
  psi = sparse(y*x/2) ;
  if param.verbose
    fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
            x, y, full(psi(1)), full(psi(2))) ;
  end
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  
  for i in 1:length(param.patterns)
      if y == param.labels{i}
          delta = 0 ; else delta = 1; end
      
      if dot(param.labels{i}*x, model.w) - dot(y*x, model.w) > 1
          yhat = y ; else yhat = - y ; end
  if param.verbose
    fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
            model.w, x, y, yhat) ;
  end
end
