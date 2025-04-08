clc;
clear;
% path='/media/iiau/win7_file/kyq/3/mat4/';
%path = './';
path = 'D:\PycharmProjects\CarDD_release\CarDD_SOD\CarDD-TE\results\';
dirpath=dir([path '*.mat']);
%dirpath{0} = dir([path '(12w_cwm).mat']);
%dirpath{1} = dir([path '(18w_ori).mat']);
%str=zeros(23,5);
str=['m','r','g','y','c','b','k','m','r','g','y','c','b','k'];%'--b','--c','--k','--y','--m'
%str=['m','r','g','y','k','c','b','--b'];
%aa=[10,11,1,6,22,23,14,15,18,19];
% % str=['r','r','b','b','g','g','k','k','m','m'];
% str=['r','b','g','k','m']; %������ɫ
rr=[];

% for i=1:length(dirpath)
%     %  load([path num2str(aa(i)) '-filter.mat']);
%     load([path dirpath(i).name]);
% 
%     plot(mRecall,mPre,str(i),'linewidth',2);
% 
% 
%     hold on;
%     display([dirpath(i).name(1:end-4)])% '----' num2str(a)]);
%     aa(i)=AUC;
%     rr=[rr;mRecall;mPre];
%     %  dirpath(i).name
%     %  display(num2str(max(pM)));
% end
% xlabel('Recall');
% ylabel('Precision');
% 
% legend(...
%     dirpath(1).name(1:end-4));
% 
% grid
% save pr AUC;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(dirpath)
    %  load([path num2str(aa(i)) '-filter.mat']);
    load([path dirpath(i).name]);

    % 对回召率和精度值进行平滑处理
    % smoothedRecall = smooth(mRecall, 0.1, 'moving'); % 0.1 是平滑因子
    % smoothedPrecision = smooth(mPre, 0.1, 'moving');

    % 或者，你也可以使用插值来增加数据点
    newRecall = linspace(min(mRecall), max(mRecall), 1000); % 500个插值点
    newPrecision = interp1(mRecall, mPre, newRecall, 'pchip'); % 使用分段三次插值法

    % 绘制平滑后的数据
    plot(newRecall, newPrecision, str(i), 'LineWidth', 2);

    % plot(mRecall,mPre,str(i),'linewidth',2);


    hold on;
    display([dirpath(i).name(1:end-4)])% '----' num2str(a)]);
    aa(i)=AUC;
    rr = [rr; newRecall; newPrecision]; % 存储平滑后的数据
    % rr=[rr;mRecall;mPre];
    %  dirpath(i).name
    %  display(num2str(max(pM)));
end
xlabel('Recall');
ylabel('Precision');

legend(...
    dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4));
%,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4)),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4);%,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4));%,dirpath(4).name(1:end-4))%,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4))
%,dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4)
%,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4)
%dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4));,dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4),dirpath(13).name(1:end-4)
%,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4)
%legend(...,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4)
%dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),,dirpath(8).name(1:end-4),,dirpath(7).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4));  ,dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4)%���߱�ע
%legend(...
%dirpath(1).name(1:end-4));  %���߱�ע
%dirpath(13).name(1:end-4),dirpath(14).name(1:end-4),dirpath(15).name(1:end-4));
%,dirpath(16).name(1:end-4));
%dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4));

% dirpath(17).name(1:end-4)...
%);

grid
save pr AUC;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:length(dirpath)
%     %  load([path num2str(aa(i)) '-filter.mat']);
%     load([path dirpath(i).name]);
%     if i==1
%         plot(mRecall,mPre,str(i),'linewidth',3);
%     else
%         plot(mRecall,mPre,str(i),'linewidth',2);
%     end
% 
%     hold on;
%     display([dirpath(i).name(3:end-4)])% '----' num2str(a)]);
%     aa(i)=AUC;
%     rr=[rr;mRecall;mPre];
%     %  dirpath(i).name
%     %  display(num2str(max(pM)));
% end
% xlabel('Recall');
% ylabel('Precision');
% 
% legend(...
%     dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4)),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4);%,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4));
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4))%,dirpath(4).name(1:end-4))
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4))
% 
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4))
% %dirpath(1).name(1:end-4));%,dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4));
% %,dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4)
% %,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4)
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4));,dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4),dirpath(13).name(1:end-4)
% %,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4)
% %legend(...,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4)
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),,dirpath(8).name(1:end-4),,dirpath(7).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4));  ,dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4)%���߱�ע
% %legend(...
% %dirpath(1).name(1:end-4));  %���߱�ע
% %dirpath(13).name(1:end-4),dirpath(14).name(1:end-4),dirpath(15).name(1:end-4));
% %,dirpath(16).name(1:end-4));
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4));
% 
% % dirpath(17).name(1:end-4)...
% % );
% 
% grid
% save pr AUC;
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:length(dirpath)
%     %  load([path num2str(aa(i)) '-filter.mat']);
%     load([path dirpath(i).name]);
%     if mod(i,2)
%         plot(mRecall,mPre,str(i),'linewidth',2.5);
%     else
%         plot(mRecall,mPre,[str(i) '-'],'linewidth',1.5);
%     end
% 
%     hold on;
%     display([dirpath(i).name(3:end-4)])% '----' num2str(a)]);
%     aa(i)=AUC;
%     rr=[rr;mRecall;mPre];
%     %  dirpath(i).name
%     %  display(num2str(max(pM)));
% end
% xlabel('Recall');
% ylabel('Precision');
% 
% legend(...
%     dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4)),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4);%,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4));
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4))%,dirpath(4).name(1:end-4))
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4))
% 
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4));
% %dirpath(1).name(1:end-4));%,dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4));
% %,dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4)
% %,dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4)
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4));,dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4),dirpath(13).name(1:end-4)
% %,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4),dirpath(10).name(1:end-4)
% %legend(...,dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4),dirpath(9).name(1:end-4)
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),,dirpath(8).name(1:end-4),,dirpath(7).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4));  ,dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4),dirpath(10).name(1:end-4),dirpath(11).name(1:end-4),dirpath(12).name(1:end-4)%���߱�ע
% %legend(...
% %dirpath(1).name(1:end-4));  %���߱�ע
% %dirpath(13).name(1:end-4),dirpath(14).name(1:end-4),dirpath(15).name(1:end-4));
% %,dirpath(16).name(1:end-4));
% %dirpath(1).name(1:end-4),dirpath(2).name(1:end-4),dirpath(3).name(1:end-4),dirpath(4).name(1:end-4),dirpath(5).name(1:end-4),dirpath(6).name(1:end-4),dirpath(7).name(1:end-4),dirpath(8).name(1:end-4));
% 
% % dirpath(17).name(1:end-4)...
% % );
% 
% grid
% save pr AUC;

