function [flag,resid1,resid2,rank1,rank2,time1,time2,nv,ps] ...
                   = CompleteZ(m,n,r,problem,opts, ...
                                tolerrank)

    Zorig     = problem.Zorig;
    Zpart     = problem.Zpart;
    indsZ     = problem.indsZ;        % sampled - positions in Zorig
    mn        = m+n;
    minsize   = opts.minsize;         % clique size
    maxsize   = opts.maxsize;         % clique size
    verbose   = opts.verbose;         % how much output
    Yfinalm   = zeros(m);
    Yfinaln   = zeros(n);

    if ~exist('tolerrank','var')
        tolerrank =  max(m,n)*eps(normest(Zpart)); 
    end

    starttic = cputime;
    
    %%%%%% forming original adjacency matrix A %%%%%%  
    Z = spalloc(m,n,length(indsZ));
    Z(indsZ) = 1;
    indskeep = find([Z;sparse(n,n)])+(m+n)*m;  % upper-triangular positions in A
    A=[(true(m))   Z
         Z'     (true(n))];
    A(eye(mn)==1) = 0;  % set diag to 0
    
    %%%%%% finding cliques %%%%%%
    ticcliques = tic;
    y = GrowCliques(A,maxsize,minsize);  % list of cliques (type: cell)
    numCliques = length(y);
    if verbose
         fprintf('time for finding %i cliques is %g \n',...
             numCliques,toc(ticcliques))
    end
  
    %%%%%% memory allocation %%%%%%
    expvctrsr = cell(numCliques,1);   % save exposing vectors rows
    expvctrsc = cell(numCliques,1);   % save exposing vectors cols
    expposr = cell(numCliques,1);     % save indices/positions rows
    expposc = cell(numCliques,1);     % save indices/positions cols
    if (noisy)
            expwtsr = zeros(numCliques,1);    % wts rows
            expwtsc = zeros(numCliques,1);    % wts cols
    end
    successr = 0;
    successc = 0;
    maxc = 0;                       % maxsize of the cliques
    
    %%%%%% main loop for exposing vectors %%%%%%
    for ii=1:numCliques        
        indsY = y{ii};                % indics in the clique
        rowsi = logical(indsY<=m);
        rowsi = indsY(rowsi);         % rows of Zorig
        colsj = logical(indsY>m);
        colsj = indsY(colsj)-m;       % cols of Zorig
        lr = length(rowsi);
        lc = length(colsj);

        if  ~((max(indsY)<=m) || (min(indsY)>=m+1)) ...
                              &&  min(lr,lc)>=r &&  max(lr,lc)>r
            if (noisy)
                pqm = min(lr,lc);
            end
            X = full(Zpart(rowsi,colsj));
            [uXtemp,sXtemp,vXtemp] = svd(X);
            diagsX = diag(sXtemp);
            rankX = find(diagsX >= tolerrank*diagsX(1), 1, 'last');
            
            if rankX < r  % should never happen generically
               fprintf('rankX < r - should not happen generically \n');
               keyboard
            end
            
            % finding the largest clique size
            if lr > r || lc > r
                maxc = max(maxc,lr+lc);
            end
            
            % finding row exposing vector
            if lr>r     
                successr=successr+1;
                if (noisy)
                    expwtsr(ii)=sum(diagsX(r+1:pqm).^2)/(.5*lr*(lr-1));
                end
                expposr{ii}=indsY(1:lr);
                expvctrsr{ii}=uXtemp(:,r+1:lr);  % save only vectors
            end
            
            % finding col exposing vector
            if lc>r
                successc=successc+1;
                if (noisy)
                    expwtsc(ii)=sum(diagsX(r+1:pqm).^2)/(.5*lc*(lc-1));
                end
                expposc{ii}=indsY(lr+1:end);
                expvctrsc{ii}=vXtemp(:,r+1:lc);  % save only vectors
            end
        end
    end
  
    if verbose
        fprintf('maxc =  %i; and %i and %i # successful row and col cliques, resp.  \n',...
                        maxc,successr,successc);
    end
    


    %%%%%% forming final exposing vector Yfinal NOISELESS case %%%%%%
    for jj=1:length(expvctrsr)  % add up exp. vctrs for cols
        clique=expposr{jj};
        if ~isempty(clique)
            temp=expvctrsr{jj}*expvctrsr{jj}';
            temp=(temp+temp')/2;         
            Yfinalm(clique,clique)= Yfinalm(clique,clique) + temp; 
        end
    end
    for jj=1:length(expvctrsc)  % add up exp. vctrs for cols
        clique=expposc{jj}-m;
        if ~isempty(clique)
            temp=expvctrsc{jj}*expvctrsc{jj}';
            temp=(temp+temp')/2;         
            Yfinaln(clique,clique)= Yfinaln(clique,clique) + temp; 
        end
    end
  
    %%%%%% checking if Yfinal blocks have 0 on the diagonal %%%%%%
    indszeroYr=find(diag(Yfinalm)==0);
    indszeroYc=find(diag(Yfinaln)==0);
    indszeroY= union(indszeroYr,(indszeroYc+m));
    if ~isempty(indszeroYr)
        if verbose
           fprintf('WARNING:  number of zero rows in Yr  %i >0  , ',...
	                 length(indszeroYr)) 
           fprintf(' shifting out diagonal \n')
        end
        indsdiagYr=sub2ind(size(Yfinalm),indszeroYr,indszeroYr);
        Yfinalm(indsdiagYr)=1e1*(rand(length(indsdiagYr),1)+1); % separate eigs
    end
    if ~isempty(indszeroYc)
        if verbose
           fprintf('WARNING:  number of zero rows in Yc  %i >0  , ',...
	                 length(indszeroYc)) 
           fprintf(' shifting out diagonal \n')
        end
        indsdiagYc = sub2ind(size(Yfinaln),indszeroYc,indszeroYc);
        Yfinaln(indsdiagYc) = 1e1*(rand(length(indsdiagYc),1)+1); % separate eigs
    end

  
    ticnull=tic;
    UYm=null(Yfinalm);
    UYn=null(Yfinaln);
    if verbose
      fprintf('time for null of Yfinal %g \n',toc(ticnull))
    end

  
    %%%% finding the final exposing vector %%%%
    rnp = size(UYm,2);
    rnq = size(UYn,2);
    V = [ [UYm ; sparse(n,rnp)]  [ sparse(m,rnq) ; UYn ]  ]; 
    nv = size(V,2);
    if verbose
      fprintf('size of V is %g %g   \n',size(V))
    end
  
    HZ = true(size(Z)); % for non sampled elements if needed
    ps = 100;           % percent sampled to return in function
    if ~isempty(indszeroY) % zero rows in Yfinal - remove rows/cols of Z data
       indszeroZrows = find(indszeroY <= m);
       if ~isempty(indszeroZrows)
           indszeroZrows = indszeroY(indszeroZrows);
	       Z(indszeroZrows,:) = 0;         % ignore rows not sampled enough
	       HZ(indszeroZrows,:) = false;    % ignore rows not sampled enough
       end
       indszeroZcols = find(indszeroY>m);
       if ~isempty(indszeroZcols)
               indszeroZcols = indszeroY(indszeroZcols)-m;
	       Z(:,indszeroZcols) = 0;         % ignore cols not sampled enough
	       HZ(:,indszeroZcols) = false;    % ignore cols not sampled enough
       end
       ps = 100*sum(sum(HZ))/numel(HZ);
       indskeep = find([Z;sparse(n,n)])+(m+n)*m;
     
       % redo adjacency due to new zeros in Z
       A=[(true(m))   Z
            Z'     (true(n))];
       A(eye(mn)==1)=0;  % set diag to 0
     
    end  % end of if for zero rows in Yfinal
  
    [indsi,indsj] = ind2sub(size(A),indskeep);
    indsi2 = indsi;
    indsj2 = indsj-m;
    matrepszz = zeros(length(indskeep),rnp*rnq);    % top right block only
  
    E2 = zeros(nv);
    E2(1:rnp,rnp+1:end) = 1;            % size of top right block of final FR R
    [mi2,mj2] = find(triu(E2,1));       %  subs for upperblock of smaller R
    indsblku = sub2ind(size(E2),mi2,mj2);

    %%% forming the mapping matrix %%%
    UYmT = UYm';   % short version
    UYnT = UYn';   % short version
    sqrt2 = sqrt(2);
    for zz=1:length(indskeep)
	    Ri2=UYmT(:,indsi2(zz));
	    Rj2=UYnT(:,indsj2(zz));
	    Toffdiag2 = (Ri2(mi2).*Rj2(mj2-rnp))/sqrt2;
	    matrepszz(zz,:)=Toffdiag2; % don't need zero columns - just rt blk
    end
  
    tempY=[sparse(m,m) Zpart;sparse(n,m+n)];
    allz=tempY(indskeep); % vector of known entries of Z
  
    [~,indssub,matrepssubzz,~] = lindep(matrepszz',tolerrank);
    matrepssubzz = matrepssubzz';
    allzsub = allz(indssub);
    if rank(matrepssubzz,1e-9)<min(size(matrepssubzz))
        fprintf('WARNING: error rank cvx too small \n')
    end

    warning off
    cvx_clear

    if ~verbose
        cvx_quiet true
    end

    cvx_begin sdp
    cvx_precision best
    variable R(nv,nv) symmetric
    minimize trace(R)
    subject to
      matrepssubzz*localsvecupblck(R,indsblku) == allzsub;
      R >= 0
    cvx_end
    warning on

    flag = 0;
    YY = V*R*V';
    YY = (YY+YY')/2; % numerically ensure symmetry
    newZ = YY(1:m,m+1:end);
    ranknewZ = rank(full(newZ));    % using default rank here
    ranknewZorig = ranknewZ;
    time1 = cputime - starttic;
    rank1 = ranknewZorig;
    resid1 = norm(HZ.*(newZ-Zorig))/norm(HZ.*Zorig);
    time2 = time1;
    rank2 = rank1;
    resid2 = resid1;
  
end % end of function completeZ


function [r,idx,Xsub,Xnull] = lindep(X,tol,trgtr)
% lindep extracts a linearly independent set of columns of a given matrix X
% in:
%   X = input matrix
%   tol = A rank estimation tolerance, default=1e-10
% out:
%   Xsub = extracted columns of X
%   idx =  indices (into X) of the extracted columns 

    if ~nnz(X) % X has no non-zeros and hence no independent columns
        Xsub=[]; 
        idx=[];
        return
    end
    
    if nargin<2, tol=1e-10; end
    
    [Q, R, E] = qr(X,0);
    if ~isvector(R)
        diagr = abs(diag(R));
    else
        diagr = R(1);
    end
    
    r = find(diagr >= tol*diagr(1), 1, 'last'); % rank estimation
    
    if nargin >2
	       r = max(r,size(X,2)-3*trgtr);   % ensure nullspace not too large
    end
    
    if nargout > 1
        idx = sort(E(1:r));
        idx = idx(:);
    end
    
    if nargout > 2
        Xsub = X(:,idx);                      
    end
    
    idx = sort(E(1:r)); 
    Xnull = Q(:,r+1:end);
    
  end % end of function lindep