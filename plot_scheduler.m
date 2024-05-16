close all; clear; clc
fid=py.open("scheduler.obj",'rb');
data=py.pickle.load(fid);
data=cell(data);
data_lin = [double(data{5}{'linear'}); double(data{1}{'linear'})]';
data_pol = [double(data{5}{'polynomial'}); double(data{1}{'polynomial'})]';
data_sqrt = [double(data{5}{'inverse_sqrt'}); double(data{1}{'inverse_sqrt'})]';
data_cos = [double(data{5}{'cosine'}); double(data{1}{'cosine'})]';
data_cos_res = [double(data{5}{'cosine_with_restarts'}); double(data{1}{'cosine_with_restarts'})]';


center_lin = mean(data_lin,1);
center_pol = mean(data_pol,1);
center_sqrt = mean(data_sqrt,1);
center_cos = mean(data_cos,1);
center_cos_res = mean(data_cos_res,1);



figure(); hold on

h(1) = plot(center_lin(:,1), center_lin(:,2), 'o', 'Linewidth', 2, 'DisplayName','linear','Color',"#0072BD");
plot(data_lin(:,1), data_lin(:,2), '.', 'MarkerSize',10 ,'Color',"#0072BD");


h(2) = plot(center_pol(:,1), center_pol(:,2), 'o', 'Linewidth', 2, 'DisplayName','pol', 'Color',"#D95319");
plot(data_pol(:,1), data_pol(:,2), '.', 'MarkerSize',10 ,'Color',"#D95319");


h(3) = plot(center_sqrt(:,1), center_sqrt(:,2), 'o', 'Linewidth', 2 , 'DisplayName', 'sqrt', 'Color',"#EDB120");
plot(data_sqrt(:,1), data_sqrt(:,2), '.', 'MarkerSize',10 ,'Color',"#EDB120");

h(4) = plot(center_cos(:,1), center_cos(:,2), 'o', 'Linewidth', 2 , 'DisplayName', 'cos', 'Color', "#7E2F8E");
plot(data_cos(:,1), data_cos(:,2), '.', 'MarkerSize',10 ,'Color',	"#7E2F8E");

h(5) = plot(center_cos_res(:,1), center_cos_res(:,2), 'o', 'Linewidth', 2, 'DisplayName','cos\_res', 'Color',"#77AC30");
plot(data_cos_res(:,1), data_cos_res(:,2), '.', 'MarkerSize' ,10 ,'Color',"#77AC30");

legend(h)
xlabel("Best Epoch");
ylabel("Best Accuracy")
