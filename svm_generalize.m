function [ bac_raw, weight_raw, bac_perm_label, weight_perm_label, bac_perm_trial, weight_perm_trial, C_cv] = svm_generalize(s1,s2,s_test1,s_test2,norm_type,ratio_train_val,ncv,nfold,permute_label,permute_trial,nperm,C_range)

%% train linear SVM with dataset from s1 and s2, then teset on dataset s3 and s4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% procedure: 
% (1) Randomly split data into training data and test/validate data for"ncv" times, 
% for each split, do the following
% (2) find optimal C-parameter from a range of C_values, 
% (3) Use optimal C to compute the model with MC cross-validations, 
% (4) compute the model with permuted class labels (2000 random permutations)

%% inputs: 
% "s1" and "s2" are spike counts in training dataset, condition -1 (s1) and condition 1 (s2), of the form (number of trials x number of neurons)
% "s_test1" and "s_test2" are spike counts in test dataset, condition -1 (s3) and condition 1 (s4), of the form (number of trials x number of neurons)
% "ratio_train_val" is the ratio of training data/validation data, 
% "ncv" is the number of Monte-Carlo cross-validations,
% "nfold" is the number of folds for the n-fold cross-validation in the training set,
% "permute_label" :permutation for the regular model, 0: no permute, 1 permute lable, 
% "permute_trial" :permutation for the regular model, 0: no permute, 1 permute trials for each neuron (remove noise correlation)
% "nperm" is the number of permutations
% "C_range" is the vector of tested C-parameters

%% outputs:
% "bac_raw" is the balanced accuracy,  [ncv by 1]
% "weight_raw" is the weight of SVM model, [ncv by n_neurons]

% "bac_perm_label" is the balanced accuracy for models with permuted labels, [ncv by nperm]
% weight_perm_label is the weight for models with permuted labels,, [ncv by nperm by n_neurons]

% "bac_perm_trial" is the balanced accuracy for models with permuted trials, [ncv by nperm]
% weight_perm_trial is the weight for models with permuted trials, [ncv by nperm by n_neurons]

% "C_cv"  is the optimal C found for each cross-validations, [ncv by 1]
% "SVMModel" is the SVM model, for each cross-validation, {1 by number of cros-validations}, 

n_neuron=size(s1,2);
bac_raw=nan(ncv,1);
weight_raw=nan(ncv,n_neuron);

bac_perm_label=nan(ncv,nperm);
weight_perm_label=nan(ncv,nperm,n_neuron);

bac_perm_trial=nan(ncv,nperm);
weight_perm_trial=nan(ncv,nperm,n_neuron);

C_cv=zeros(ncv,1);
SVMModel=cell(1,ncv);
%% 
n1=size(s1,1); % number of total trials for condition 1
n2=size(s2,1); % number of total trials for condition 2
ntrain1=floor(n1*ratio_train_val); % number of trials for training from condition 1
ntrain2=floor(n2*ratio_train_val); % number of trials for training from condition 2

label_train=cat(1,(-1)*ones(ntrain1,1),ones(ntrain2,1));              % labels training

ntest1=size(s_test1,1); % number of total trials for condition 1
ntest2=size(s_test2,1); % number of total trials for condition 2
label_test=cat(1,(-1)*ones(ntest1,1),ones(ntest2,1));              % labels test

N=floor(length(label_train)/nfold); % number of samples in n-fold cross-validation

%% classify
parfor idx_cv=1:ncv
    fprintf(['cross validation repeat',num2str(idx_cv),'/',num2str(ncv),'\n'])    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (1) permute trial order for Monte-Carlo cross-validation for random splits in training and test set 
    rp1=randperm(n1);
    rp2=randperm(n2);
    
    s1_train=s1(rp1(1:ntrain1),:); % take samples for training; condition 1
    s2_train=s2(rp2(1:ntrain2),:); % take samples for training; condition 2    
    s_train=cat(1,s1_train,s2_train); % concatenate training data in condition 1 & condition 2
    s_train_mean=mean(s_train); % use the mean and std from training dataset to normalize the training data and test data
    s_train_std=std(s_train);    
    s_train_norm=(s_train-repmat(s_train_mean,size(s_train,1),1))./repmat(s_train_std,size(s_train,1),1);
    s_train_norm(isnan(s_train_norm))=0;
    if norm_type==1
        data_train=s_train_norm;   
    else
        data_train=s_train;
    end

    s_test=cat(1,s_test1,s_test2);     % concatenate validation data in condition 1 and 2 
    s_test_norm=(s_test-repmat(s_train_mean,size(s_test,1),1))./repmat(s_train_std,size(s_test,1),1);
    s_test_norm(isnan(s_test_norm))=0;   
    if norm_type==1
        data_test=s_test_norm;
    else
        data_test=s_test;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (2) Using training data to find the optimal C-parameter. 
    % The search for the optimal C-parameter is done for every Monte Carlo split loop
    % The training set is split into 10 folds; the model is trained on 9 folds and tested on the remaining fold.
    % This is done for every C parameter. After averaging across 10-folds, we choose the C parameter that gave the model with best performance. 
    
    rp_train=randperm(size(data_train,1)); % permute order of trials for the n-fold cv 
    x_train_new=data_train(rp_train,:); 
    label_train_new=label_train(rp_train);        % use the same order for labels
    
    bac_c=zeros(length(C_range),nfold);
    for idx_c=1:length(C_range)                % range of C-parameters        
        for idx_fold=1:nfold           
            x_train_c_fold=[x_train_new(1:(idx_fold-1)*N,:);x_train_new(idx_fold*N + 1 : end,:)];       % data for training
            lable_train_c_fold=[label_train_new(1:(idx_fold-1)*N);label_train_new(idx_fold * N + 1:end)]; % label training            
            x_val_c_fold=x_train_new(1+(idx_fold-1)*N:idx_fold*N,:);                              % data validation
            lable_val_c_fold=label_train_new(1+(idx_fold-1)*N:idx_fold*N);                          % label validation           
            try
                svmstruct=fitcsvm(x_train_c_fold, lable_train_c_fold,'KernelFunction','linear','CacheSize','maximal','BoxConstraint',C_range(idx_c));
                lable_predicted_c_fold=predict(svmstruct,x_val_c_fold);                
                tp =length(find(lable_val_c_fold==1 & lable_predicted_c_fold==1)); % TruePos
                tn =length(find(lable_val_c_fold==-1 & lable_predicted_c_fold==-1)); % TrueNeg
                fp =length(find(lable_val_c_fold==-1 & lable_predicted_c_fold==1)); % FalsePos
                fn =length(find(lable_val_c_fold==1 & lable_predicted_c_fold==-1)); % FalseNeg              
                % compute balanced accuracy
                if (tn+fp)==0
                    bac_c(idx_c,idx_fold) =tp./(tp+fn); % to avoid NaN
                elseif (tp+fn)==0
                    bac_c(idx_c,idx_fold) =tn./(tn+fp); % to avoid NaN
                else
                    bac_c(idx_c,idx_fold) =((tp./(tp+fn))+(tn./(tn+fp)))./2;
                end
            catch
                bac_c(idx_c,idx_fold)=nan;                
            end           
        end
    end    
    [~,idx]=nanmax(nanmean(bac_c,2)); % average across 10-fold cv and take the max across the tested C-parameters
    C=C_range(idx);                % choose the regularization parameter with highest accuracy
    C_cv(idx_cv)=C;  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (3) train and test the model 
    try
        SVMModel{idx_cv}=fitcsvm(data_train, label_train,'KernelFunction','linear','CacheSize','maximal','BoxConstraint',C);        
        lable_predicted=predict(SVMModel{idx_cv},data_test); % validate
        
        % get balanced accuracy
        tp =length(find(label_test==1 & lable_predicted==1)); % TruePos
        tn =length(find(label_test==-1 & lable_predicted==-1)); % TrueNeg
        fp =length(find(label_test==-1 & lable_predicted==1)); % FalsePos
        fn =length(find(label_test==1 & lable_predicted==-1)); % FalseNeg        
        if (tn+fp)==0
            bac_raw(idx_cv)=tp./(tp+fn);
        elseif (tp+fn)==0
            bac_raw(idx_cv)=tn./(tn+fp);
        else
            bac_raw(idx_cv)=((tp./(tp+fn))+(tn./(tn+fp)))./2;
        end

        if size(SVMModel{idx_cv}.SupportVectors,1)>=1
            % svmmdl.Alpha~ Lagrange multiplier
            % svmmdl.SupportVectors~(number of support vectors/used trails,n_neurons)
            w_temp=repmat(SVMModel{idx_cv}.Alpha.*SVMModel{idx_cv}.SupportVectorLabels,1,n_neuron).*SVMModel{idx_cv}.SupportVectors;
            weight_raw(idx_cv,:)=sum(w_temp,1);
        end        
    catch
        bac_raw(idx_cv)=nan;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (4) train and test the same data with permutation of labels
    if permute_label==1
        bac_perm=nan(1,nperm);
        weight_perm_i=nan(nperm,size(s1,2));
        for idx_perm=1:nperm
            label_train_perm=label_train(randperm(length(label_train)));                                                                              % permutation of  training labels
            try               
                SVMModel_perm=fitcsvm(data_train, label_train_perm,'KernelFunction','linear','CacheSize','maximal','BoxConstraint',C);        
                lable_predicted=predict(SVMModel_perm,data_test); % validate
                               
                tp =length(find(label_test==1 & lable_predicted==1)); % TruePos
                tn =length(find(label_test==-1 & lable_predicted==-1)); % TrueNeg
                fp =length(find(label_test==-1 & lable_predicted==1)); % FalsePos
                fn =length(find(label_test==1 & lable_predicted==-1)); % FalseNeg               
                if (tn+fp)==0
                    bac_perm(idx_perm)=tp./(tp+fn);
                elseif (tp+fn)==0
                    bac_perm(idx_perm)=tn./(tn+fp);
                else
                    bac_perm(idx_perm)=((tp./(tp+fn))+(tn./(tn+fp)))./2;
                end
                if size(SVMModel_perm.SupportVectors,1)>=1
                    w_temp=repmat(SVMModel_perm.Alpha.*SVMModel_perm.SupportVectorLabels,1,n_neuron).*SVMModel_perm.SupportVectors;
                    weight_perm_i(idx_perm,:)=sum(w_temp,1);
                end
            catch
                bac_perm(idx_perm)=nan;
            end
        end
        bac_perm_label(idx_cv,:)=bac_perm;
        weight_perm_label(idx_cv,:,:)=weight_perm_i;
    end
    if permute_trial==1
        bac_perm=nan(1,nperm);
        weight_perm_i=nan(nperm,n_neuron);
        s1_train_norm=data_train(1:ntrain1,:);
        s2_train_norm=data_train(ntrain1+1:end,:);
        s1_train_norm_perm=nan(size(s1_train_norm));
        s2_train_norm_perm=nan(size(s2_train_norm));        
        for idx_perm=1:nperm
            for idx_neuron=1:n_neuron
                s1_train_norm_perm(:,idx_neuron)=s1_train_norm(randperm(ntrain1),idx_neuron); % permutation of data    
                s2_train_norm_perm(:,idx_neuron)=s2_train_norm(randperm(ntrain2),idx_neuron); % permutation of data                      
            end
            data_train_perm=cat(1,s1_train_norm_perm,s2_train_norm_perm);
            try               
                SVMModel_perm=fitcsvm(data_train_perm,label_train,'KernelFunction','linear','CacheSize','maximal','BoxConstraint',C);        
                lable_predicted=predict(SVMModel_perm,data_test); % validate
                               
                tp =length(find(label_test==1 & lable_predicted==1)); % TruePos
                tn =length(find(label_test==-1 & lable_predicted==-1)); % TrueNeg
                fp =length(find(label_test==-1 & lable_predicted==1)); % FalsePos
                fn =length(find(label_test==1 & lable_predicted==-1)); % FalseNeg               
                if (tn+fp)==0
                    bac_perm(idx_perm)=tp./(tp+fn);
                elseif (tp+fn)==0
                    bac_perm(idx_perm)=tn./(tn+fp);
                else
                    bac_perm(idx_perm)=((tp./(tp+fn))+(tn./(tn+fp)))./2;
                end
                if size(SVMModel_perm.SupportVectors,1)>=1
                    w_temp=repmat(SVMModel_perm.Alpha.*SVMModel_perm.SupportVectorLabels,1,size(SVMModel_perm.SupportVectors,2)).*SVMModel_perm.SupportVectors;
                    weight_perm_i(idx_perm,:)=sum(w_temp,1);
                end
            catch
                bac_perm(idx_perm)=nan;
            end
        end
        bac_perm_trial(idx_cv,:)=bac_perm;
        weight_perm_trial(idx_cv,:,:)=weight_perm_i;        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
end
%%
end

