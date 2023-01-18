clear;
addpath(genpath("mesh2d"));

% Origami ribbon

%% Input parameters
wribbon = 0.8;
lribbon = 6;
lbonding = 1.2;
meshsize = 0.3;

%% Generate mesh
x1 = -lribbon/2 - lbonding;
x2 = -lribbon/2;
x3 = lribbon/2;
x4 = lribbon/2 + lbonding;
y1 = -lbonding/2;
y2 = -wribbon/2;
y3 = wribbon/2;
y4 = lbonding/2;
points = [x1, y1; x2, y1; x2, y2; x3, y2; x3, y1; x4, y1; x4, y4; x3, y4; x3, y3; x2, y3; x2, y4; x1, y4];
pairs = [1,2; 2,3; 3,4; 4,5; 5,6; 6,7; 7,8; 8,9; 9,10; 10,11; 11,12; 12,1];
parts = {[1:12]};
[P, ~, T, tnum] = refine2(points, pairs, parts, [], meshsize);
P(:,3) = 0;
NP = size(P, 1);
NT = size(T, 1);

% Correct node orders
P1 = P(T(:,1),:);
P2 = P(T(:,2),:);
P3 = P(T(:,3),:);
Area = P1(:,1).*P2(:,2) - P2(:,1).*P1(:,2) + P2(:,1).*P3(:,2) ...
    - P3(:,1).*P2(:,2) + P3(:,1).*P1(:,2) - P1(:,1).*P3(:,2);
Area = Area/2;
I = 1:NT;
Ineg = I(Area<0);
if ~isempty(Ineg)
   tmp = T(Ineg,1);
   T(Ineg,1) = T(Ineg,2);
   T(Ineg,2) = tmp;
end

% Regions
Nregion = [2, 1];
region = ones(NP,1);
% Fixed nodes
region(P(:,1)<=-lribbon/2) = -1;
region(P(:,1)>=lribbon/2-0.03) = -2;

figure()
hold on
patch('faces',T(:,1:3),'vertices',P,'facecolor','none','edgecolor', [.7,.7,.7]);
scatter(P(:,1), P(:,2), 20, region, 'filled');
colormap(parula(sum(Nregion)+1));
colorbar();
daspect([1 1 1]);


% Write node position file
fid = fopen('coords.txt', 'w');
fprintf(fid,'%.19f %.19f %.19f \n',[P']);
fclose(fid);

% Write region lists
fid = fopen('regions.txt', 'w');
fprintf(fid,'%i \n',[region']);
fclose(fid);

% Write triangulation
fid = fopen('tlist.txt', 'w');
fprintf(fid,'%i %i %i \n',[T'-1]);
fclose(fid);
