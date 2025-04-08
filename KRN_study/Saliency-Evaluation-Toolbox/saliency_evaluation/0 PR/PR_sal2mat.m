%% PR
clc;clear;close all;

% set the ground truth path
gt_dir = { 'CarDD', 'D:\PycharmProjects\CarDD_release\CarDD_SOD\CarDD-TE\CarDD-TE-Mask\',[],[] 'png' };

% set the save path
basedir = 'D:\PycharmProjects\CarDD_release\CarDD_SOD\CarDD-TE\results\';
%%
% mkdir( basedir );

alg_dir = ...                                           
{  
% set the saliency map path as well as name.
 { 'SAM2.1-UNet', 'D:\PycharmProjects\CarDD_release\CarDD_SOD\CarDD-TE\results\MFFN\test_mask', [],'', 'png' };  %1. RGBD results\MPCI
% {'Ours', '/home/wlz/Downloads/caffe-future/models/finetune1/Experiments/ECSSD/map_Refcn-filter/', [],'' 'jpg'};
};

alg_dir_FF = candidateAlgStructure( alg_dir );  
dataset = datasetStructure( gt_dir(1), gt_dir(2) );

[ mPre, mRecall, mFmeasure, mHitRate , mFalseAlarm, AUC ] = ...
    performCalcu(dataset,alg_dir_FF); 


save( [ basedir 'MFFN'], 'mPre', 'mRecall', 'mFmeasure', 'mHitRate', 'mFalseAlarm', 'AUC' );

curve_bar_plot( basedir, gt_dir, alg_dir, mPre, mRecall, mFmeasure, AUC );

