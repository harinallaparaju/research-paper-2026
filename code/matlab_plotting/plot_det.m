% DET Curve Plotter matching Figure 6
clc; clear; close all;

data = readtable('../unimodal_implementation/2002_1/results.csv');
fmr = data.FAR; % FMR is conceptually FAR
fnmr = data.FRR; % FNMR is conceptually FRR

% Calculate roughly EER point
[~, eerIdx] = min(abs(fmr - fnmr));
eerVal = (fmr(eerIdx) + fnmr(eerIdx)) / 2;

fig = figure('Position', [150, 100, 600, 500]);
axes1 = axes('Parent', fig);
hold(axes1, 'on');
grid(axes1, 'on');

% Main Plot with markers
plot(axes1, fmr, fnmr, '-ob', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'none');

% EER Diagonal Line
plot(axes1, [0, 1], [0, 1], '-r', 'LineWidth', 1);

% EER Circle marking
plot(axes1, eerVal, eerVal, 'or', 'LineWidth', 3, 'MarkerSize', 8, 'MarkerFaceColor', 'none');
text(axes1, eerVal + 0.02, eerVal, sprintf('EER = %.2f', eerVal*100), ...
    'FontName', 'Times New Roman', 'FontSize', 11);

% Styling
xlabel('False Match Rate (FMR)', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('False Non-Match Rate (FNMR)', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
xlim(axes1, [0 1]);
ylim(axes1, [0 1.05]);
set(axes1, 'FontSize', 11, 'FontName', 'Times New Roman', 'LineWidth', 1);

% Legend
legend(axes1, {'FMR vs FNMR', 'EER'}, 'Location', 'northeast', 'FontSize', 11, 'FontName', 'Times New Roman');

hold(axes1, 'off');
% print('-depsc','-r300','fig6_det_curve.eps');
