close all; clear; clc
fid=py.open("lrsave.obj",'rb');
data=py.pickle.load(fid);
data=cell(data);
data_16 = [double(data{5}{1e-06}); double(data{1}{1e-06})]';
data_56 = [double(data{5}{5e-06}); double(data{1}{5e-06})]';
data_55 = [double(data{5}{5e-05}); double(data{1}{5e-05})]';
data_14 = [double(data{5}{0.0001}); double(data{1}{0.0001})]';
data_13 = [double(data{5}{0.001}); double(data{1}{0.001})]';


center_16 = mean(data_16,1);
center_56 = mean(data_56,1);
center_55 = mean(data_55,1);
center_14 = mean(data_14,1);
center_13 = mean(data_13,1);



figure(); hold on

h(1) = plot(center_16(:,1), center_16(:,2), 'o', 'Linewidth', 2, 'DisplayName','1e-6','Color',"#0072BD");
plot(data_16(:,1), data_16(:,2), '.', 'MarkerSize',10 ,'Color',"#0072BD");


h(2) = plot(center_56(:,1), center_56(:,2), 'o', 'Linewidth', 2, 'DisplayName','5e-6', 'Color',"#D95319");
plot(data_56(:,1), data_56(:,2), '.', 'MarkerSize',10 ,'Color',"#D95319");


h(3) = plot(center_55(:,1), center_55(:,2), 'o', 'Linewidth', 2 , 'DisplayName', '5e-5', 'Color',"#EDB120");
plot(data_55(:,1), data_55(:,2), '.', 'MarkerSize',10 ,'Color',"#EDB120");

h(4) = plot(center_14(:,1), center_14(:,2), 'o', 'Linewidth', 2 , 'DisplayName', '1e-4', 'Color', "#7E2F8E");
plot(data_14(:,1), data_14(:,2), '.', 'MarkerSize',10 ,'Color',	"#7E2F8E");

h(5) = plot(center_13(:,1), center_13(:,2), 'o', 'Linewidth', 2, 'DisplayName','1e-3', 'Color',"#77AC30");
plot(data_13(:,1), data_13(:,2), '.', 'MarkerSize' ,10 ,'Color',"#77AC30");

legend(h)
xlabel("Best Epoch");
ylabel("Best Accuracy")
