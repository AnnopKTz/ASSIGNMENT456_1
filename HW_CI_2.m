clear;
data = readData();
data = Normalize(data);
validation_set = cross_validation(data);

learning_rate = 0.3;
momentum_rate = 0.3;
min_err = 10.^-5;
err_avg = 1 ;
max_epoch = 1000;
crr_epoch = 1;
hidden_layers1 = 4;
hidden_layers2 = 8;


w_hi = ones(hidden_layers1,2);
w_hh = ones(hidden_layers2,hidden_layers1);
w_oh = ones(2,hidden_layers2);

w_h1b = 1:hidden_layers1;
w_h2b = 1:hidden_layers2;
w_ob = 1:2;

total_yh1 = zeros(hidden_layers1,max_epoch);
total_yh2 = zeros(hidden_layers2,max_epoch);
total_yo = zeros(2,max_epoch);

total_err = zeros(2,max_epoch,10);
total_confustion = zeros(4,10);
total_acc = zeros(1,10);


for i = 1:10
    test_data = validation_set(:,:,i);
    train_data = AddDataTrain(validation_set,i);
    [w_hi,w_hh,w_oh,w_h1b,w_h2b,w_ob] = GenerateW(w_hi,w_hh,w_oh,w_h1b,w_h2b,w_ob);
    %train
    while(crr_epoch <= max_epoch)
        data_rand = randperm(size(train_data,1));
        
        %set all weigth
        if crr_epoch == 1
            [total_w_hi,total_w_hh,total_w_oh,total_w_h1b,total_w_h2b,total_w_ob]= SetFirstWeigth(max_epoch,w_hi,w_hh,w_oh,w_h1b,w_h2b,w_ob);
        end
        
        for j = 1 : length(data_rand)
            %-----------------------Feed forward
            dj = train_data(data_rand(j),3:4);
            vk = 1 : hidden_layers1;
            for k = 1 : hidden_layers1
                vk = 0;
                for n =1: 2
                    vk =  vk + ((train_data(data_rand(j),n)*total_w_hi(k,n,crr_epoch)));
                end
                vk = vk + total_w_h1b(k,crr_epoch);
                total_yh1(k,crr_epoch) = Sigmoid(vk);
            end
            
            for k = 1 : hidden_layers2
                vk = 0;
                for n = 1 : hidden_layers1
                    vk =  vk + (total_w_hh(k,n,crr_epoch)*total_yh1(n,crr_epoch));
                end
                vk = vk + total_w_h2b(k,crr_epoch);
                total_yh2(k,crr_epoch) = Sigmoid(vk);
            end
            
            for k = 1 : 2
                yo = 0;
                for n = 1 : hidden_layers2
                    yo = yo + (total_yh2(n,crr_epoch)*total_w_oh(k,n,crr_epoch));
                end
                yo = yo + total_w_ob(k,crr_epoch);
                total_yo(k,crr_epoch) = Sigmoid(yo);
                total_err(k,crr_epoch,i)= (dj(k) - total_yo(k,crr_epoch));
            end
            
            
            %-----------------------back propagation
            %gradian output
            gd_out = 1:2;
            gd_out(1) = DiffSigmoid(total_yo(1,crr_epoch))*total_err(k,crr_epoch,i);
            gd_out(2) = DiffSigmoid(total_yo(1,crr_epoch))*total_err(k,crr_epoch,i);
            
            %w_oh
            for n = 1 : 2
                for k = 1 : hidden_layers2
                    if crr_epoch == 1
                        delta_w_oh = (learning_rate*gd_out(n)*total_w_oh(n,k,crr_epoch));
                    else
                        delta_w_oh = (learning_rate*gd_out(n)*total_w_oh(n,k,crr_epoch))+ (momentum_rate*((total_w_oh(n,k,crr_epoch))-total_w_oh(n,k,crr_epoch-1)));
                    end
                    total_w_oh(n,k,crr_epoch+1) = total_w_oh(n,k,crr_epoch) + delta_w_oh;
                end
                
                %w_ob
                if crr_epoch == 1
                    delta_w_ob = (learning_rate*gd_out(n)*total_w_ob(n,crr_epoch));
                else
                    delta_w_ob = (learning_rate*gd_out(n)*total_w_ob(n,crr_epoch))+(momentum_rate *((total_w_ob(n,crr_epoch)-total_w_ob(n,crr_epoch-1))));
                end
                
                total_w_ob(n,crr_epoch+1) = total_w_ob(n,crr_epoch) + delta_w_ob;
                
            end


            gd_h2 = zeros(1, hidden_layers2);
            for n = 1 : hidden_layers2
                for k = 1 : 2
                    gd_h2(n) =  (gd_out(k)* total_w_oh(k,n,crr_epoch))+ gd_h2(n);
                end
                gd_h2(n) = DiffSigmoid(total_yh2(n,crr_epoch))*gd_h2(n);
                
                for k = 1 : hidden_layers1
                     if crr_epoch == 1
                        delta_w_hh = (learning_rate*gd_h2(n)*total_w_hh(n,k,crr_epoch));
                    else
                        delta_w_hh = (learning_rate*gd_h2(n)*total_w_hh(n,k,crr_epoch))+ (momentum_rate*((total_w_hh(n,k,crr_epoch))-total_w_hh(n,k,crr_epoch-1)));
                    end
                    total_w_hh(n,k,crr_epoch+1) = total_w_hh(n,k,crr_epoch) + delta_w_hh;
                end
                
                %w_h2b
                if crr_epoch == 1
                    delta_w_h2b = (learning_rate*gd_h2(n)*total_w_h2b(n,crr_epoch));
                else
                    delta_w_h2b = (learning_rate*gd_h2(n)*total_w_h2b(n,crr_epoch))+(momentum_rate *((total_w_h2b(n,crr_epoch)-total_w_h2b(n,crr_epoch-1))));
                end
                
                total_w_h2b(n,crr_epoch+1) = total_w_h2b(n,crr_epoch) + delta_w_h2b;
            end
            
            
            gd_h1 = zeros(1, hidden_layers1);
            for n = 1 : hidden_layers1
                for k = 1 : hidden_layers2
                    gd_h1(n) =  (gd_h2(k)* total_w_hh(k,n,crr_epoch))+ gd_h1(n);
                end
                gd_h1(n) = DiffSigmoid(total_yh1(n,crr_epoch))*gd_h1(n);
                
                for k = 1 : 2
                     if crr_epoch == 1
                        delta_w_hi = (learning_rate*gd_h1(n)*total_w_hi(n,k,crr_epoch));
                    else
                        delta_w_hi = (learning_rate*gd_h1(n)*total_w_hi(n,k,crr_epoch))+ (momentum_rate*((total_w_hi(n,k,crr_epoch))-total_w_hi(n,k,crr_epoch-1)));
                    end
                    total_w_hi(n,k,crr_epoch+1) = total_w_hi(n,k,crr_epoch) + delta_w_hi;
                end
                
                %w_h1b
                if crr_epoch == 1
                    delta_w_h1b = (learning_rate*gd_h1(n)*total_w_h1b(n,crr_epoch));
                else
                    delta_w_h1b = (learning_rate*gd_h1(n)*total_w_h1b(n,crr_epoch))+(momentum_rate *((total_w_h1b(n,crr_epoch)-total_w_h1b(n,crr_epoch-1))));
                end
                
                total_w_h1b(n,crr_epoch+1) = total_w_h1b(n,crr_epoch) + delta_w_h1b;
            end
            

        end
        fprintf('THIS EPOCH : %d of validation_set %d\n',crr_epoch,i);
        crr_epoch = crr_epoch+1;
    end
    
    
    %test confusion matrix
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    for j = 1: size(test_data,1)
        yk1 = 1 : hidden_layers1;
        yk2 = 1 : hidden_layers2;
        
        for k = 1 : hidden_layers1
            yk1(k) = 0;
            for n =1: 2
                yk1(k) =  yk1(k) + (test_data(j,n)*total_w_hi(k,n,max_epoch+1));
            end
            yk1(k) = yk1(k) + total_w_h1b(k,1001);
        end
        
        for k = 1 : hidden_layers2
            yk2(k) = 0;
            for n =1: hidden_layers1
                yk2(k) =  yk2(k) + (yk1(n)*total_w_hh(k,n,max_epoch+1));
            end
            yk2(k) = yk2(k) + total_w_h2b(k,1001);
        end
        
        out = zeros(1,2);
        for k = 1:2
            out(k) = 0;
            for n = 1: hidden_layers2
                out(k) = out(k) + (yk2(k)*total_w_oh(k,n,1001));
            end
            out(k) = out(k) + total_w_ob(k,1001);
        end
       
        if out(1) > out(2)
            check = [0.9,0.1];
            if check == test_data(j,3:4)
                TP = TP + 1;
            else
                FN = FN +1;
            end
        else
            check = [0.1,0.9];
            if check == test_data(j,3:4)
                TN = TN + 1;
            else
                FP = FP +1;
            end
        end
        total_confustion(1,i) = TP;
        total_confustion(2,i) = FP;
        total_confustion(3,i) = TN;
        total_confustion(4,i) = FN;
    end
    
    total_acc(i) = (TP+TN)/ (TP+TN+FP+FN) ;
    crr_epoch = 1;
end



%part function
function out = readData()
    data = fopen('cross.pat');
    text_line = fgetl(data);
    out = zeros(1,4);
    index = 1;
    line = 1;
    while ischar(text_line)
        x = str2double(split(text_line));
        if mod(index,3)==2
            out(line,1) = x(1,1);
            out(line,2) = x(2,1);
        elseif mod(index,3)==0
            out(line,3) = x(1,1);
            out(line,4) = x(2,1);
            line = line+1;
        end
        index = index +1;
        text_line = fgetl(data);
    end
    fclose(data);
end

function data = Normalize(input)
    in_vec = reshape(input,[],1);
    nor_in = normalize(in_vec,'range',[0.1,0.9]);
    data = reshape(nor_in,[size(input,1),size(input,2)]);
end

function data_train = cross_validation(x)
    x_rand = randperm(size(x,1));
    size_validation = fix(size(x,1)/10);
    data_train = zeros(size_validation,4,10);
    index = 1;
    in_i = 1;
    in_j = 1;
    for i = 1 : size(x,1)
        for j = 1 : size(x,2)
            data_train(in_i,in_j,index) = x(x_rand(i),j);
            in_j =in_j +1;
        end
        in_i = in_i +1;
        in_j = 1;
        if mod(i,size_validation) == 0
            index = index +1;
            in_i = 1;
        end
    end
end

function sigm = Sigmoid(n)
    sigm = 1.0 / (1.0 + exp(-n));
end

function result = DiffSigmoid(n)
    result = n * (1-(n));
end

function data = AddDataTrain(s,n)
    data = zeros((size(s,1)*9+4),size(s,2));
    index = 1;
    for i = 1 : size(s,3)
        if i ~= n && i ~= 11
            for j = 1 : size(s,1)
                for k = 1 : size(s,2)
                    data(index,k) = s(j,k,i);
                end
                index = index + 1;
            end
        elseif i ==11
            for j = 1 : 4
                for k = 1 : size(s,2)
                    data(index,k) = s(j,k,i);
                end
                index = index + 1;
            end
        end
    end
end

function [w1,w2,w3,w4,w5,w6] = GenerateW(a,b,c,d,e,f)
    w_rand_range = (-1:0.001: 1);
    w_rand_index = randperm(length(w_rand_range));
    
    for i = 1 : size(a,1)
        for j = 1:2
            w1(i,j) = w_rand_range(w_rand_index(1));
            w_rand_index(1)=[];
        end
    end
    
    for i = 1 : size(b,1)
        for j = 1 : size(b,2)
            w2(i,j) = w_rand_range(w_rand_index(1));
            w_rand_index(1)=[];
        end
    end
    
    for i = 1 : size(c,1)
        for j = 1 : size(c,2)
            w3(i,j) = w_rand_range(w_rand_index(1));
            w_rand_index(1)=[];
        end
    end
    
    for i = 1 : length(d)
        w4(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
    for i = 1 : length(e)
        w5(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
    for i = 1 : length(f)
        w6(i) = w_rand_range(w_rand_index(1));
        w_rand_index(1)=[];
    end
    
end

function [w1,w2,w3,w4,w5,w6]= SetFirstWeigth(m,a,b,c,d,e,f)
    w1 = zeros(size(a,1),size(a,2),m);
    w2 = zeros(size(b,1),size(b,2),m);
    w3 = zeros(size(c,1),size(c,2),m);
    w4 = zeros(size(d,2),m);
    w5 = zeros(size(e,2),m);
    w6 = zeros(size(f,2),m);
    
    w1(:,:,1) = a;
    w2(:,:,1) = b;
    w3(:,:,1) = c;
    w4(:,1) = d;
    w5(:,1) = e;
    w6(:,1) = f;
end