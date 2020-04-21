%% Read data
data = readtable('mall_customers.csv');
%% Preprocess data
data_p = zeros(200,5); %Variable that stores raw data as array for MATLAB processing
data_p(:,3) = table2array(data(:,3),'Age'); 
data_p(:,4) = table2array(data(:,4),'AnnualIncome_k__');
data_p(:,5) = table2array(data(:,5),'SpendingScore_1_100_');

for i = 1:200
    if strcmp(table2array(data(i,2),'Gender'),'Male')
        data_p(i,1) = 1; %Male column will display 1 if male
        data_p(i,2) = 0; %Female column will display 0 if male
    else
        data_p(i,1) = 0;
        data_p(i,2) = 1;
    end
end
writematrix(data_p,'data.csv','Delimiter',',') %convert preprocessed data to csv for Python script
%% Post Processing
%% Read the optimum cluster generated label file
label = readtable('labels_gmm.csv');
label = table2array(label);
%% Separate data belonging to different clusters in different variables
 hs1 = data_p(label==0,:); %store one cluster in one variable
 mean_hs1 = mean(hs1,1) %Find column wise mean for each cluster
 hs2 = data_p(label==1,:);
 mean_hs2 = mean(hs2,1)
 hs3 = data_p(label==2,:);
 mean_hs3 = mean(hs3,1)
 hs4 = data_p(label==3,:);
 mean_hs4 = mean(hs4,1)
 hs5 = data_p(label==4,:);
 mean_hs5 = mean(hs5,1)
 hs6 = data_p(label==5,:);
 mean_hs6 = mean(hs6,1)
 %% Find standard deviation of cluster with highest mean of spending score
 std(hs1,1)