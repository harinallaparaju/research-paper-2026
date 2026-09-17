% ROC Curve Plotter in exact style from the journal
% Reads results.csv and computes AUC with shading
clc; clear; close all;

% 1. Load Data
% Make sure your python script outputs a CSV with FAR and TAR (or GAR)
% For demonstration, we load realistic curve data
data = readtable('../unimodal_implementation/2002_1/results.csv');
far = data.FAR; 
tar = 1 - data.FRR; % True Acceptance Rate (or 1 - False Rejection Rate)

% Sort for interpolation
[far, sortIdx] = sort(far);
tar = tar(sortIdx);

% Calculate AUC using trapezoidal numerical integration
auc = trapz(far, tar);

% 2. Setup Figure
fig = figure('Position', [100, 100, 600, 500]);
axes1 = axes('Parent', fig);
hold(axes1, 'on');
grid(axes1, 'on');

% 3. Area Shading
fill(axes1, [far; 1; 0], [tar; 0; 0], [0.8 1 0.8], 'EdgeColor', 'none'); 

% 4. Master Plot lines
plot(axes1, far, tar, '-ob', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'none');
plot(axes1, [0, 1], [0, 1], '--r', 'LineWidth', 1.5); % Equal Error Line / Base

% 5. Styling to match image
xlabel('FAR', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('TAR', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
title('', 'FontName', 'Times New Roman');

% Adjusted axis limits 
xlim(axes1, [0 1]);
ylim(axes1, [0 1.05]);
set(axes1, 'FontSize', 11, 'FontName', 'Times New Roman', 'LineWidth', 1);

% Legend
lgdStr = sprintf('ROC (AUC = %.4f)', auc);
legend(axes1, {lgdStr, 'FAR=TAR'}, 'Location', 'southeast', 'FontSize', 11, 'FontName', 'Times New Roman');

hold(axes1, 'off');

% Save as high-res EPS for journal embedding
% print('-depsc','-r300','fig7_roc_curve.eps');
