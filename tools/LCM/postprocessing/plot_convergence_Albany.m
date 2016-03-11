%
% Run at prompt in Albany run folder before executing this script:
%
%    grep -i "||F||" <Albany Output File> > <Norm Output File>
%

stringPathDirectory = '<path>';
stringPathFile = '<Norm Output File>';
handleFile = fopen([stringPathDirectory, stringPathFile]);

countStep=1;
countIteration=0;

clear normF

while (~feof(handleFile))
   line = fgetl(handleFile); 
   countIteration=countIteration+1; 
   normF{countStep}(countIteration) = str2num(line(9:17)); 
   normDu{countStep}(countIteration) = str2num(line(43:51));
   if (strfind(line,'Converged'))
      countIteration=0; 
      countStep=countStep+1; 
   end
end


% Plot residual norm convergence

handleFigure = figure(114);

stepPlot = numel(normF);
%stepPlot = 2;
loglog(normF{stepPlot}(1:end-1),normF{stepPlot}(2:end), ...
   normF{stepPlot},normF{stepPlot},'k:',normF{stepPlot},normF{stepPlot}.^2,'b:')
set(gca,'FontSize',20)
xlabel('Error e_{n}')
ylabel('Error e_{n+1}')
legend('Simulation','Linear converge','Quadratic converge','orientation','horizontal','location','northoutside')

handleFigure = figure(115);
semilogy(normF{end})
set(gca,'FontSize',20)
xlabel('Iteration n')
ylabel('Error e_{n}')
legend('Simulation data','orientation','horizontal','location','northoutside')



% Plot displacment increment norm convergence
handleFigure = figure(116);

stepPlot = numel(normDu);
%stepPlot = 2;
loglog(normDu{stepPlot}(1:end-1),normDu{stepPlot}(2:end), ...
   normDu{stepPlot},normDu{stepPlot},'k:',normDu{stepPlot},normDu{stepPlot}.^2,'b:')
set(gca,'FontSize',20)
xlabel('Error e_{n}')
ylabel('Error e_{n+1}')
legend('Simulation','Linear converge','Quadratic converge','orientation','horizontal','location','northoutside')

handleFigure = figure(117);
semilogy(normDu{end})
set(gca,'FontSize',20)
xlabel('Iteration n')
ylabel('Error e_{n}')
legend('Simulation data','orientation','horizontal','location','northoutside')