% ================================================
% Script used to plot maximum temperature
%
% Author: Mario J. Juha
% Scientific Computation Research Center (SCOREC)
% Troy - NY
% 12 / 17 / 2015
% ================================================

% Specify max. number of step file to read. Assuming they are
% consecutively numerated and starting from 0.
% i.e. temperature.30.cvs
nfiles = 30;
% specify time step
dt = 1e-5;

% specify base name for file without mesh adaptation
basefile_name = 'temperature.';

% specify base name for file with mesh adaptation
basefile_name2 = 'temperatureAdapt4.';

% File extension
file_ext = '.csv';


% inizilite variables
t = zeros(1,nfiles+1);
maxTemp = zeros(1,nfiles+1);
maxTemp2 = zeros(1,nfiles+1);


% loop over files and find max. temperature
for i=0:nfiles
    p = num2str(i);
    temp_file = strcat(basefile_name,p,file_ext);
    A = csvread(temp_file,1,0);
    maxTemp(i+1) = max(A(:,2));
    %
    temp_file2 = strcat(basefile_name2,p,file_ext);
    B = csvread(temp_file2,1,0);
    maxTemp2(i+1) = max(B(:,2));
    %
    t(i+1) = i*dt;
end


% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
box(axes1,'on');
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(t,maxTemp,t,maxTemp2);
set(plot1(1),'DisplayName','No adapt');
set(plot1(2),'DisplayName','Adapt 4','LineStyle','--','Color',[1 0 0]);

% Create legend
legend1 = legend(axes1,'show');
set(legend1,'Location','best');
xlabel('Time')
ylabel('Max. Temperature')    
    