 #load data - wdbc

require(Matrix)

###Solve WLS problem

#simulate data

# create sparse matrix with simulated data
n = 1000
p = 500 
X = matrix(rnorm(n*p), nrow=n)
mask = matrix(rbinom(n*p,1,0.04), nrow=n, ncol=p)
X = mask*X
beta = runif(p)
y = X %*% beta + rnorm( n, mean = 0, sd = 1)
W <- diag(rep(1, n)) 

#inversion method
inversion <- function(y,X,W)
{
  return(solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% y)
}

#QR decomposition method
QRdec <- function(y,X,W)
{
  betahat <- 1
  
  #decomposition
  Wsqrt = diag(sqrt(diag(W)))
  QR = qr(diag(sqrt(diag(W)))%*%X)
  
  #solve R*betahat = t(Q)*Wsqrt
  QW = t(qr.Q(QR)) %*% W.sqrt %*% y
  R = qr.R(QR)            #components of decomposition
  for(j in ncol(X):1){
    index = c(2:ncol(X),0)[j:ncol(X)]
    betahat[j] = (QW[j] - sum(R[j,index]*betahat[index]))/R[j,j]
  }
  return(betahat)
}

#sparse matrix

###Gradient descent problem

#Predictor variables X ~ first 10 features
X <- as.matrix(wdbc[,c(3,12)])
#Add ones to X
X <- cbind(rep(1, nrow(X)),X)
#scale X for computability
X = scale(X)

#Response variable ~ classification as M (1) or B (0) - I modified the data file
y <- as.matrix(wdbc[,2])
m <- nrow(y)

###
#Gradient descent
###

#get gradient of fn given weights
grad.get <- function(y, X, w, m) {
  grad <- -t(X) %*% (y - m*w)	#gradient of logisitic fn
  return(grad)
}

#GD algorithm
#alpha = learning rate (~.01 is good)
#iter = maximum number of iterations (~100)
grad.descent <- function(X, y, Binit, iter, alpha, tol){
  #set up data structures
  N = length(y)
  p = ncol(x)
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  max.iter = iter + 1
  loglikelihood = rep(0,max.iter)
  distance = rep(0,max.iter)
  ss = rep(1,N)
  
  for (i in 2:max.iter) {
    w = as.numeric(1 / (1 + exp(-X %*% Betas[i-1,])))
    Betas[i,] <- Betas[i-1,] - alpha  * grad.get(X, y, w, ss)   #the key step in GD
    distance[i] = sqrt(sum(Betas[i,] - Betas[i-1,])^2) #distance is sum of squared differences
    if(distance[i] < tol)  #break and return latest Betas if distance is less and tolerance
    {
      return (Betas)
      print("distance: " + distance)
      break
    }
    
    #update negative loglikelihood
    newloglikelihood = -log(w+.00001)*y - log(1-w+.00001)*(m-y)
    loglikelihood[i] =  newloglikelihood
  }
  return(Betas)
}

#
# Gradient descent with backtracking line search
#

#alphamax > 0
#alpha1 e(0,alphamax)
#rho ~.6
grad.descent.BLS <- function(X, y, Binit, iter, rho, c, tol)
{
  #initialize data structures
  N = length(y)
  p = ncol(x)
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  max.iter = iter + 1
  ll = rep(0,max.iter)
  distance = rep(0,max.iter)
  ss = rep(1,N)
  
  
  for (i in 2:max.iter) {
    #initial alpha and w
    alpha = 1
    w = as.numeric(1 / (1 + exp(-X %*% Betas[i-1,])))
    #descent direction
    pt = -grad.get(y, X, w, ss)
    #old log likelihood
    llOLD = -log(w+.00001)*y - log(1-w+.00001)*(ss-y)
    
    while(llNEW > llOLD)
    {
      Betas[i,] = Betas[i-1,] + alpha*pt
      wNEW = as.numeric(1 / (1 + exp(-X %*% Betas[i,])))
      llNEW = -log(wNEW+.00001)*y - log(1-wNEW+.00001)*(ss-y)
      llOLD = llOLD + .001*alpha*crossprod(-pk, pk)
      alpha = alpha*rho
    }
    
    distance[i] = sqrt(sum(Betas[i,] - Betas[i-1,])^2) #distance is sum of squared differences
    if(distance[i] < tol)  #break and return latest Betas if distance is less and tolerance
    {
      return (Betas)
      print("distance: " + distance)
      break
    }
    
  }
  return(Betas)
}

#
#Stochastic gradient descent - constant alpha
#
sgd1 <- function(X, y, Binit, iter, alpha, tol)
{
  #set up data structures
  N = length(y)
  p = ncol(x)
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  max.iter = iter + 1
  loglikelihood = rep(0,max.iter)
  distance = rep(0,max.iter)
  ss = rep(1,N)
  
  for (i in 2:max.iter) {
    #get random sample data point
    ran = sample(1:N, 1)
    xran = X[ran]
    yran = y[ran]
    
    #get gradient
    grad <- -(yran-xran*Betas[i-1,])*xran ###?
    #SCD step based on gradient
    Betas[i,] <- Betas[i-1,] - alpha*grad   #stochastic GD step
    
    distance[i] = sqrt(sum(Betas[i,] - Betas[i-1,])^2) #distance is sum of squared differences
    if(distance[i] < tol)  #break and return latest Betas if distance is less and tolerance
    {
      return (Betas)
      print("distance: " + distance)
      break
    }
    
    #update negative loglikelihood
    newloglikelihood = -log(w+.00001)*y - log(1-w+.00001)*(m-y)
    loglikelihood[i] =  newloglikelihood
  }
  print("Iterations:" + iter)
  return(Betas)
}

#
#Stochastic gradient descent - decaying alpha a la Robbins-Munro
#

#LR (learning rate), C, and t0 are part of Robbins-Monro rule
#C > 0 
#LR e[.05,1], closer to one means faster decay
#t0 ~ 1,2

sgd2<- function(X, y, Binit, iter, LR, C, t0, tol)
{
  #set up data structures
  N = length(y)
  p = ncol(x)
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  max.iter = iter + 1
  loglikelihood = rep(0,max.iter)
  distance = rep(0,max.iter)
  ss = rep(1,N)
  
  for (i in 2:max.iter) {
    #get random sample data point
    ran = sample(1:N, 1)
    xran = X[ran]
    yran = y[ran]
    
    #get gradient based on data point
    grad <- -(yran-xran*Betas[i-1,])*xran ###?
    #get alpha based on Robbins-Monro rule
    alpha = (C*(i+t0))^(-LR)
    #SCD step based on gradient
    Betas[i,] <- Betas[i-1,] - alpha*grad 
    
    distance[i] = sqrt(sum(Betas[i,] - Betas[i-1,])^2) #distance is sum of squared differences
    if(distance[i] < tol)  #break and return latest Betas if distance is less and tolerance
    {
      #time averaged Betas
      BetasTimeAvg <- (Betas*i + Betas)/(i+1)
      return (Betas)
      return (BetasTimeAvg)
      print("distance: " + distance)
      break
    }
    
    #update negative loglikelihood
    newloglikelihood = -log(w+.00001)*y - log(1-w+.00001)*(m-y)
    loglikelihood[i] =  newloglikelihood
  }
  print("Iterations:" + iter)
  return(Betas)
  return(BetasTimeAvg)
}

#modify BLS to get adaptive gradient
AdaGrad <- function(X, y, Binit, iter, alpha, tol) #rho, c
{
  #initialize data structures
  N = length(y)
  p = ncol(x)
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  max.iter = iter + 1
  ll = rep(0,max.iter) #keep track of log likelihood values
  distance = rep(0,max.iter)
  ss = rep(1,N) #vector of sample sizes
  
  #initialize hessian estimate
  hessian_approx = matrix(0.001, ncol(X))
  #set fudge factor
  fudgefactor = .000001
  
  for (i in 2:max.iter) {
    #get current data point
    Xcurrent = X[i]
    Ycurrent = y[i]
    Mcurrent = ss[i]
    
    #get gradient & hessian, update betas based on these using Adagrad formula
    grad = -(Ycurrent-Xcurrent*Betas[i-1,])*Xcurrent
    hessian = hessian_approx + grad^2
    Betas[i,] <- Betas[i-1,] - alpha*grad / sqrt(hessian_approx+fudgefactor)
    
    #update log likelihoods
    
  }
  
  return(Betas)
}
###

#
#Newton's method
#

#set initial parameters
iter = 100 #number of iterations
tol = .001 #tolerance

newton = function(X, y, Binit, tol, m, iter)
{
  #initialize data structurces
  N = length(y)
  p = ncol(x)
  Betas = matrix(0,length(Binit), p)
  Betas[1,] = Binit
  max.iter = iter + 1
  loglikelihood = rep(0,max.iter)
  distance = rep(0,max.iter)
  ss = rep(1,N)
  
  for (i in 2:max.iter)
  {
    w = as.numeric(1 / (1 + exp(-X %*% Betas[i-1,]))) #calculate weights
    H = hessian(X,ss,w) #calculate Hessian
    G = grad.get(y, X, w, ss) #calculate gradient 
    
    #solve Hx = G
    solve(H,G)
    
    #find Beta step using Cholesky decompositions
    u = solve(t(chol(H))) %*% G
    v = solve(chol(H)) %*% u
    
    #add new Betas to Beta matrix
    Betas[i,] = Beta[i,] + v
    
    #find distance between latest Betas and previous Betas
    distance[i] = sqrt((sum((Betas[i,] - Betas[i-1,])^2))) #distance is sum of squared differences
    if(distance[i] < tol)  #break and return latest Betas if distance is less and tolerance
    {
      return (Betas)
      return (BetasTimeAvg)
      print("distance: " + distance)
      break
    }
    
    #update negative loglikelihood
    newloglikelihood = -log(w+.00001)*y - log(1-w+.00001)*(m-y)
    loglikelihood[i] =  newloglikelihood
    
  }
  return (Betas)
  return (BetasTimeAvg)
  print (Betas)
}

#Quasi-Newton 
QN = function(X, y, Binit, tol, m, iter)
{
  
}