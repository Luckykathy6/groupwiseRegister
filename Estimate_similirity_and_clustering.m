function Estimate_similirity_and_clustering(datapath,scan_list,codepath)

% This function aims to calculate similarity index between each pair of FA/MD images after rigid body transformation and cluster them into different subgroups using Louvain clustering 
% The clustering results is visualized and saved out as a fig file
% Kuaikuai Duan, Longchuan Li, Marcus Autism Center, 03/11/2024

%INPUTS:    
%datapath: the data path where all subjects' FA and MD maps in NIFTI format are located
%scan_list: a text file with the list of scans to be used for clustering
%codepath: path where all scripts are saved

% OUTPUTS: 
%subgroups, each subgroup has a text file listing the included subjects/scans

addpath(genpath(fullfile(codepath,'2019_03_03_BCT'))) %%add brin connectivity toolbox to the path for using 'community_louvain.m' and 'grid_communities.m'

%read FA, MD data
subjid=textread(scan_list,'%s',1000);

for i=1:length(subjid)
    img=[];
    ind1 = strfind(subjid{i},'.nii.gz');
    if ~isnan(ind1)
    	[img, dims,scales,bpp,endian] = read_avw([datapath,'/',subjid{i}(1:ind1-1),'_aff_fa.nii.gz']);
    else
	[img, dims,scales,bpp,endian] = read_avw([datapath,'/',subjid{i},'_aff_fa.nii.gz']);
    end
    sz=size(img);
    img_all_fa(:,i)=reshape(img,[sz(1)*sz(2)*sz(3),1]);
    % calculate volume and mean FA;
    idx=find(img>0);
    brain_vol(i)=length(idx)*8/1000;
    brain_fa(i)=mean(img(idx));
    
    img=[];
    ind2 = strfind(subjid{i},'.nii.gz');
    if ~isnan(ind1)
    	[img, dims,scales,bpp,endian] = read_avw([datapath,'/',subjid{i}(1:ind2-1),'_aff_tr.nii.gz']);
    else
	[img, dims,scales,bpp,endian] = read_avw([datapath,'/',subjid{i},'_aff_tr.nii.gz']);
    end
    img_all_tr(:,i)=reshape(img,[sz(1)*sz(2)*sz(3),1]);
    disp(['finished loading subject: ',subjid{i}]);  
end

%Compute the similarity between each FA/MD image pairs
dist_mat_fa=zeros(length(subjid),length(subjid));
dist_mat_tr=dist_mat_fa;
for i=1:size(img_all_fa,2)
    for j=i+1:size(img_all_fa,2);
        dist_mat_fa(i,j)=sum(abs(img_all_fa(:,i)-img_all_fa(:,j)).^2,1)/size(img_all_fa,1);
        dist_mat_tr(i,j)=sum(abs(img_all_tr(:,i)-img_all_tr(:,j)).^2,1)/size(img_all_tr,1);
    end
end
dist_mat_fa = dist_mat_fa' + dist_mat_fa;
dist_mat_tr = dist_mat_tr' + dist_mat_tr;
    
dist_inv_fa=1./dist_mat_fa;
inf_idx_fa=isinf(dist_inv_fa); dist_inv_fa(inf_idx_fa)=0;
dist_inv_fa=(dist_inv_fa - min(dist_inv_fa(:)))./max(dist_inv_fa(:));

dist_inv_tr=1./dist_mat_tr;
inf_idx_tr=isinf(dist_inv_tr); dist_inv_tr(inf_idx_tr)=0;
dist_inv_tr=(dist_inv_tr - min(dist_inv_tr(:)))./max(dist_inv_tr(:));
AIJ=(dist_inv_fa + dist_inv_tr)/2;

%%%Clustering the similarity matrix using the louvin algorithm
gamma=1;
[M, Q1] = community_louvain_iterative(AIJ, gamma);
[X Y INDSORT] = grid_communities(M); %%%% Use 'grid_communities.m' in brain connectivity toolbox
figure;                         
imagesc(AIJ(INDSORT,INDSORT)); colormap(gca, jet(256)); colorbar(gca);   % plot ordered adjacency matrix % Ignore pink map and use jet instead. clims
hold on;                                        % hold on to overlay community visualization
plot(X,Y,'k','linewidth',2);                    % plot community boundaries
hold off;
saveas(gcf,'Louvain_clustering_results.fig');

%%%% write out each subgroup as a separate text file
fspec='%20s\n';
for i=1:max(unique(M))
    fid=fopen([datapath,'/subjid_grp',num2str(i),'.txt'],'w');
    for j=1:length(subjid)
        if M(j)==i
            fprintf(fid,fspec, subjid{j});
        end
    end
    fclose(fid);
end

disp('leaving Estimate_similirity_and_clustering...');


function [M,Q1]=community_louvain_iterative(W,gm)

M_all=[]; Q_all=[];
n  = size(W,1);             % number of nodes

M=ones(1,n); % initial community affiliations
Q0 = -1; Q1 = 0;
i=1; % initialize modularity values
while Q1-Q0>1e-9;           % while modularity increases
    Q0 = Q1;
    [M, Q1] = community_louvain(W, gm); % perform community detection using Louvain clustering in brain connectivity toolbox
    i=i+1;
end

