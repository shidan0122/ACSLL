function [ P, H, W ] = ACSLL( X, class_num, V, param )
%% initialization
[n, d] = size(X.fea);
P = rand(d, class_num);
W = ones(V, n)*(1/V);
H = mapminmax(rand(n, class_num),0,1);
Gamma = diag(ones(d,1));
eps = 0.001;
% epsil = 0.01;
niter = 20;
% % =====================   Normalization =====================
% for i = 1 :V
%     for  j = 1:n
%         X.data{i}(j,:) = ( X.data{i}(j,:) - mean( X.data{i}(j,:) ) ) / std( X.data{i}(j,:) ) ;
%     end
% end
for v = 1:V
    dv{v} = size(X.data{v},2);
    temp1 = initfcm(class_num,n);
    H_d.data{v} = temp1';
    O{v} = rand(dv{v},class_num);
end
%% optimization
% objvalue = zeros(1,niter);
for q = 1:niter
    % update O, Hv
    for v = 1:V
        for i = 1:n
            aaa{v} = zeros(1,dv{v});
            for j = 1:class_num
                aaa{v} = aaa{v} + H_d.data{v}(i, j) * X.data{v}(i, :);
                O{v}(:, j) = aaa{v} / sum(H_d.data{v}(:, j));
                distance{v}(i, j) = mydist(X.data{v}(i, :)', O{v}(:, j));
                distance{v}(i, j) = distance{v}(i, j)^2;
            end
            ad{v} = (param.beta*W(v, i)*H(i, :)-distance{v}(i,:)/(2*param.alpha)) / (1+param.beta*W(v, i)'*W(v, i))*1e-5;%
            H_d.data{v}(i,:) =  EProjSimplex_new(ad{v});
        end
    end
    % H
    for j = 1:n
        formulation_part = zeros(class_num,1);
        for i = 1:V
            formulation_part = formulation_part + W(i,j)*H_d.data{v}(j,:)';
        end
        formulation_part1 = formulation_part + param.lambda*P'*(X.fea(j,:)');
        formulation_part11 = formulation_part1';
        H(j,:) = EProjSimplex_new(formulation_part11);
    end
    % W
    for j = 1:n
        %         B_j = zeros(class_num,V);
        for v = 1:V
            B_j(:,v) = H(j,:)-H_d.data{v}(j,:);
        end
        W(:,j) = ((B_j'*B_j+eye(v)*1e-4)\ones(V,1))/(ones(V,1)'/(B_j'*B_j+eye(v)*1e-4)*ones(V,1));
    end
    % P
    while 1
        value_P1 = norm(X.fea*P - H,'fro') + param.gamma*trace(P'*Gamma*P);
        temp_gamma = zeros(d,1);
        for i = 1:d
            temp_gamma(i) = 1/(2*sqrt(P(i,:)*P(i,:)')+eps);
        end
        Gamma  = diag(temp_gamma);
        P = (X.fea'*X.fea + param.gamma*Gamma)\X.fea'*H;
        value_P2 = norm(X.fea*P - H,'fro') + param.gamma*trace(P'*Gamma*P);
        if abs(value_P1-value_P2)<eps
            break;
        end
    end
    
    %     %% compute objective function value2
    %      temp_xoh11 = 0;
    %     temp_xoh12 = 0;
    %     for v = 1:V
    %         temp_dist{v} = zeros(1);
    %         for i = 1:n
    %             for j = 1:class_num
    %                 distance{v}(i, j) = mydist(X.data{v}(i, :)', O{v}(:, j));
    %                 distance{v}(i, j) = distance{v}(i, j)^2;
    %                 temp111 = distance{v}(i,j)*H_d.data{v}(i,j);
    %                 temp_dist{v} = temp_dist{v} + distance{v}(i,j)*H_d.data{v}(i,j); %
    %             end
    %         end
    %         temp_xoh11 = temp_xoh11 + temp_dist{v}; %
    %         temp_xoh12 = temp_xoh12 + norm(H_d.data{v},'fro')^2;
    %     end
    %     temp_formulation1 = temp_xoh11 + param.alpha*temp_xoh12;
    %
    %     temp_formulation2 = 0;
    %     for i = 1:n
    %         temp_H_i = zeros(1,class_num);
    %         for v = 1:V
    %             temp_H_j = temp_H_i + W(v,i)*H_d.data{v}(i,:);
    %         end
    %         temp_formulation2 = temp_formulation2 + norm(H(i,:)-temp_H_j,'fro')^2;
    %     end
    %     temp_formulation31 = norm(X.fea*P - H,'fro')^2;
    %     temp_formulation32 = 0;
    %     for i = 1:d
    %         temp_formulation32 = temp_formulation32 + norm(P(i,:),2);
    %     end
    %     temp_formulation3 = temp_formulation31 + param.gamma*temp_formulation32;
    %     value2 = temp_formulation1 + param.beta*temp_formulation2 + param.lambda*temp_formulation3;
    %
    %     num = num + 1;
    %     %% convergence condition
    %     if(abs(value1-value2)<epsil)
    %         break;
    %     end
    %     disp(['iter:',num2str(q),',objvalues:',num2str(value1),',part1:',num2str(temp_formulation1),',part2:',num2str(temp_formulation2),',part3:',num2str(temp_formulation3)]);
end
end



