% Ce programme est le script principal permettant d'illustrer
% un algorithme de reconnaissance de chiffres.

% Nettoyage de l'espace de travail
clear all; close all;

% Repertories contenant les donnees et leurs lectures
addpath('Data');
addpath('Utils')

rng('shuffle')

precApprox = 0.75;

% Bruit
sig0=0.02;

%tableau des csores de classification
% intialisation aléatoire pour affichage
r=rand(6,5);
r2=rand(6,5);

for k=1:5
% Definition des donnees
file=['D' num2str(k)]

% Recuperation des donnees
disp('Generation de la base de donnees');
sD=load(file);
D=sD.(file);
%

% Bruitage des données
Db= D+sig0*rand(size(D));


%%%%%%%%%%%%%%%%%%%%%%%
% Analyse des donnees 
%%%%%%%%%%%%%%%%%%%%%%%
disp('PCA : calcul du sous-espace');
%%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
disp('TO DO')
[m,n] = size(Db);
individu_moyen = (Db * ones(n,1)) / n;
Db_centree = Db - individu_moyen * ones(1,n);
sigma = (1/n)*Db_centree*(Db_centree');
[W,D]=eig(sigma);
[ValPropre,tri] = sort(diag(D),'descend');
U = W(:,tri);

k_approx = 1;
val = 0;
while val < precApprox
 val = 1 - sqrt(ValPropre(k_approx)/ValPropre(1));
 k_approx = k_approx + 1;
end

C = U(:,1:k_approx)' * Db_centree;


%U = U(:,1:k_approx);
%Y = U' * Db_centree;
%%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%

disp('kernel PCA : calcul du sous-espace');
%%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
disp('TO DO')
K0 = zeros(n,n);
for i=1:n
    for j=1:n
        K0(i,j) = noyau(Db_centree(:,i),Db_centree(:,j));
    end
end
oneDivN=ones(n,n)/n;
K=K0-oneDivN*K0-K0*oneDivN+oneDivN*K0*oneDivN;
[W2,D2]=eig(K);
[ValPropre2,tri2] = sort(diag(D2),'descend');
U2 = W2(:,tri2);

k_approx2 = 1;
val2 = 0;
while val2 < precApprox && k_approx2 < n
 val2 = 1 - sqrt(ValPropre2(k_approx2)/ValPropre2(1));
 k_approx2 = k_approx2 + 1;
end

alpha = zeros(n,k_approx2);
for l=1:k_approx2
    alpha(:,l) = U2(:,l)/sqrt(ValPropre2(l));
end


%%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconnaissance de chiffres
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % Lecture des chiffres à reconnaitre
 disp('test des chiffres :');
 tes(:,1) = importerIm('test1.jpg',1,1,16,16);
 tes(:,2) = importerIm('test2.jpg',1,1,16,16);
 tes(:,3) = importerIm('test3.jpg',1,1,16,16);
 tes(:,4) = importerIm('test4.jpg',1,1,16,16);
 tes(:,5) = importerIm('test5.jpg',1,1,16,16);
 tes(:,6) = importerIm('test9.jpg',1,1,16,16);

 
 for tests=1:6
    % Bruitage
    tes(:,tests)=tes(:,tests)+sig0*rand(length(tes(:,tests)),1);
    
    % Classification depuis ACP
     %%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
     disp('PCA : classification');
     disp('TO DO')
     C_image = U(:,1:k_approx)'*(tes(:,tests) - individu_moyen);
     r(tests,k) = norm((eye(size(Db,1)) - U(:,1:k_approx)*U(:,1:k_approx)')...
     *(tes(:,tests) - individu_moyen),2)/ norm((tes(:,tests) - individu_moyen),2);
     if(tests==k)
       figure(100+k)
       subplot(1,3,1); 
       imshow(reshape(tes(:,tests),[16,16]));
       title('Image');
       subplot(1,3,2);
       imshow(reshape(individu_moyen+U(:,1:k_approx)*C_image,[16,16]));
       title('ACP');
     end  
    %%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%
  
   % Classification depuis kernel ACP
     %%%%%%%%%%%%%%%%%%%%%%%%% TO DO %%%%%%%%%%%%%%%%%%
     disp('kernel PCA : classification');
     disp('TO DO')
     x = tes(:,tests) - individu_moyen;

     %calcul de la distnace : d = 1 - nominatuer/denominateur
    
     %calcul du dénominateur (norme de (phi(x) - m )^2)
     sum1 = 0;
     for s=1:n 
         sum1 = sum1 + noyau(Db_centree(:,s),x);
     end
     denominateur = noyau(x,x) - 2*sum1/n + sum(K0*ones(n,1))/(n^2);

     

     %calcul du numérateur (la norme de la projection de phi(x) sur le sous-espace engendre par la famillede vecteurs de  H)
      
     beta = zeros(k_approx2,1);  %(les  beta(i) sont les coefficients associe a la decomposition de proj(phi(x)-m) sur la base  vk)
     for i=1:k_approx2
         sum2 =0;
         for j=1:n
             sum2 = sum2 + alpha(j,i)*noyau(x,Db_centree(:,j));
         end
         sum3 =0;
         for l=1:n
            for j=1:n
                sum3 = sum3 + alpha(l,i)*K0(j,l)/n;
            end
         end
         beta(i) = sum2-sum3;
     end

     numerateur = 0;
     for i=1:k_approx2
        for j=1:k_approx2
            for p=1:n
                for q=1:n
                    numerateur = numerateur + beta(i)*beta(j)*alpha(p,i)*alpha(q,j)*K0(p,q);
                end
            end
        end
     end
     r2(tests,k) = 1 - numerateur/denominateur;
    if(tests==k)
        %uniqement pour le noyau gaussien
        z = zeros(m,1);
        for iter = 1:4
           num = zeros(m,1);
           denom = 0;
            for i = 1:n
                gamma = 0;
               for j = 1:k_approx2
                  for l = 1:n
                      gamma = gamma + alpha(l,j)*alpha(i,j)*noyau(x,Db_centree(:,l));
                  end
                end
                num = num + gamma*noyau(Db_centree(:,i),z)*Db(:,i);
                denom = denom + gamma*noyau(Db_centree(:,i),z);
            end
            z = num/denom;
        end
      
     figure(100+k)
       subplot(1,3,3);
       imshow(reshape(z,[16,16]));
       title('Kernel ACP');
    end      
    %%%%%%%%%%%%%%%%%%%%%%%%% FIN TO DO %%%%%%%%%%%%%%%%%%    
 end
 
end


% Affichage du résultat de l'analyse par PCA
couleur = hsv(6);

figure(11)
for tests=1:6
     hold on
     plot(1:5, r(tests,:),  '+', 'Color', couleur(tests,:));
     hold off
 
     for i = 1:4
        hold on
         plot(i:0.1:(i+1),r(tests,i):(r(tests,i+1)-r(tests,i))/10:r(tests,i+1), 'Color', couleur(tests,:),'LineWidth',2)
         hold off
     end
     hold on
     if(tests==6)
       testa=9;
     else
       testa=tests;  
     end
     text(5,r(tests,5),num2str(testa));
     hold off
 end

% Affichage du résultat de l'analyse par kernel PCA
figure(12)
for tests=1:6
     hold on
     plot(1:5, r2(tests,:),  '+', 'Color', couleur(tests,:));
     hold off
 
     for i = 1:4
        hold on
         plot(i:0.1:(i+1),r2(tests,i):(r2(tests,i+1)-r2(tests,i))/10:r2(tests,i+1), 'Color', couleur(tests,:),'LineWidth',2)
         hold off
     end
     hold on
     if(tests==6)
       testa=9;
     else
       testa=tests;  
     end
     text(5,r2(tests,5),num2str(testa));
     hold off
 end

function k = noyau(x,y)
    %Noyau linéaire
    %k = x'*y;

    %Noyau polynomial
    %k = (x'*y + 1)^2;

    %noyau gaussien
   sigma = 5;
   k = exp((-(norm(x-y))^2)/(2*sigma^2));
end
