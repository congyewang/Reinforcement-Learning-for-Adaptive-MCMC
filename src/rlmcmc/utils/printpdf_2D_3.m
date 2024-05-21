function printpdf_2D_3(filename)

set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');

h = gcf;
box on
% pos = [1,1,12,4];
% pos = [1,1,12,8];
pos = [1,1,12,12];
set(h,'Units','Inches');
set(gcf,'Position',pos)
set(gcf,'color','w');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

% print(h,[filename,'.pdf'],'-dpdf','-r0')
print(h,[filename,'.pdf'],'-dpdf','-r300')
end